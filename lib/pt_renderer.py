import os

import torch
import torch.nn as nn
from pytorch3d.renderer import (
  RasterizationSettings,
  MeshRasterizer,
  SoftSilhouetteShader,
  HardFlatShader,
  SfMPerspectiveCameras,
  BlendParams,
)

from .bfm import BFM
from .Renderer import Renderer


class PtRender(nn.Module):
  def __init__(self, opt):
    super(PtRender, self).__init__()
    self.opt = opt
    self.input_res = opt.input_res
    model_path = os.path.join(os.path.dirname(__file__), '..', opt.BFM)
    self.BFM = BFM(model_path)

    f = 1015.
    self.f = f
    c = self.input_res / 2

    K = [[f,  0., c],
         [0., f,  c],
         [0., 0., 1.]]
    self.register_buffer('K', torch.FloatTensor(K))
    self.register_buffer('inv_K', torch.inverse(self.K).unsqueeze(0))
    self.K = self.K.unsqueeze(0)
    self.set_Illu_consts()

    # for pytorch3d
    self.t = torch.zeros([1, 3], dtype=torch.float32)
    self.pt = torch.zeros([1, 2], dtype=torch.float32)
    self.fl = f * 2 / self.input_res,
    ptR = [[[-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]]
    self.ptR = torch.FloatTensor(ptR)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))
    raster_settings = RasterizationSettings(
      image_size=self.input_res,
      blur_radius=0,
      faces_per_pixel=1,
      max_faces_per_bin=1000000,
    )

    # renderer
    cameras = SfMPerspectiveCameras(focal_length=self.fl,
                                    R=self.ptR.expand(opt.batch_size, -1, -1),
                                    device='cuda')
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader_rgb = HardFlatShader(blend_params=blend_params)

    self.renderer = Renderer(rasterizer, shader_rgb, SoftSilhouetteShader(), cameras)


  def forward(self, meshes):
    image, mask, fragments = self.renderer(meshes)
    image = image[..., :3]
    mask = (mask[..., -1] > 0).float()
    depth = fragments.zbuf

    return image, mask, depth

  def set_RotTransLight(self, coeff, index):
    # coeff: 33, maybe some normalization can be put here
    angles = coeff[:, :3]  # ruler angles(x,y,z) for rotation of dim 3
    self.gamma = coeff[:, 3:30]  # lighting coeff for 3 channel SH function of dim 27
    translation = coeff[:, 30:]  # translation coeff of dim 3
    # be careful for translation
    translation = self.decode_translation(translation, index)
    self.set_transform_matrices(torch.cat([angles, translation], dim=-1))

  def decode_translation(self, translation, index):
    # estimated a, b, c
    # (c + 10)(a + cx - w/2) / f, (c + 10)(b + h/2 - cy) / f, c + 10 --> t_x, t_y, t_z
    trans = torch.zeros(translation.size()).cuda()
    cx = (index % (self.input_res // 8) + 0.5) * 8
    cy = (index // (self.input_res // 8) + 0.5) * 8
    w, h = self.input_res, self.input_res
    trans[:, 0] = (translation[:, 2] + 10) * (translation[:, 0] + cx - w/2) / self.f
    trans[:, 1] = (translation[:, 2] + 10) * (translation[:, 1] + h/2 - cy) / self.f
    trans[:, 2] = translation[:, 2] + 10
    # print(trans.mean(dim=0).detach().cpu().numpy())
    return trans

  def Split_coeff(self, coeff):
    id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
    ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
    tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
    return id_coeff, ex_coeff, tex_coeff, coeff[:, 224:]

  def set_transform_matrices(self, view):
    self.rot_mat, self.trans_xyz = get_transform_matrices(view)

  def RotTrans(self, Shape, inv=False):
    # in [-1, 1] range
    if not inv:
      Shape_ = Shape.bmm(self.rot_mat.transpose(2, 1))
      Shape_[..., -1] *= -1
      Shape_ += self.trans_xyz
    else:
      Shape_ = Shape - self.trans_xyz
      Shape_[..., -1] *= -1
      Shape_ = Shape_.bmm(self.rot_mat)
    return Shape_

  def get_Landmarks(self, Shape):
    b = Shape.size(0)
    K = self.K.expand(b, 3, 3)
    projection = Shape.bmm(K.transpose(2, 1))
    projection_ = projection[..., :2] / projection[..., 2:]
    return projection_[:, self.BFM.keypoints, :]
    # return projection_

  # compute face shape with identity and expression coeff, based on BFM model
  # input: id_coeff with shape [1,80]
  #         ex_coeff with shape [1,64]
  # output: face_shape with shape [1,N,3], N is number of vertices
  def Shape_formation(self, id_coeff, ex_coeff):
    n_b = id_coeff.size(0)
    face_shape = torch.einsum('ij,aj->ai', self.BFM.idBase, id_coeff) + \
                 torch.einsum('ij,aj->ai', self.BFM.exBase, ex_coeff) + \
                 self.BFM.meanshape

    face_shape = face_shape.view(n_b, -1, 3)
    # re-center face shape
    face_shape = face_shape - self.BFM.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

    return face_shape

  # compute vertex texture(albedo) with tex_coeff
  # input: tex_coeff with shape [1,N,3]
  # output: face_texture with shape [1,N,3], RGB order, range from 0-255
  def Texture_formation(self, tex_coeff):
    n_b = tex_coeff.size(0)
    face_texture = torch.einsum('ij,aj->ai', self.BFM.texBase, tex_coeff) + self.BFM.meantex

    face_texture = face_texture.view(n_b, -1, 3)
    return face_texture

  def Illumination(self, Albedo, canoShape):
    face_norm = self.Compute_norm(canoShape)
    # face_norm = face_norm[:2]
    face_norm_r = face_norm.bmm(self.rot_mat.transpose(2, 1))
    face_color, lighting = self.Illumination_layer(Albedo, face_norm_r, self.gamma)
    return face_color, lighting

  def Compute_norm(self, face_shape):

    face_id = self.BFM.tri  # vertex index for each triangle face, with shape [F,3], F is number of faces
    point_id = self.BFM.point_buf  # adjacent face index for each vertex, with shape [N,8], N is number of vertex
    shape = face_shape
    v1 = shape[:, face_id[:, 0], :]
    v2 = shape[:, face_id[:, 1], :]
    v3 = shape[:, face_id[:, 2], :]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = e1.cross(e2, dim=2)  # compute normal for each face
    empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)

    face_norm = torch.cat((face_norm, empty), 1)  # concat face_normal with a zero vector at the end

    v_norm = face_norm[:, point_id, :].sum(2)  # compute vertex normal using one-ring neighborhood
    # CHJ: not average, directly normalize
    v_norm = v_norm / (v_norm.norm(dim=2).unsqueeze(2) + 1e-8)  # normalize normal vectors

    return v_norm

  def Illumination_layer(self, face_texture, norm, gamma):

    n_b, num_vertex, _ = face_texture.size()
    n_v_full = n_b * num_vertex
    gamma = gamma.view(-1, 3, 9).clone()
    gamma[:, :, 0] += 0.8

    gamma = gamma.permute(0, 2, 1)

    a0, a1, a2, c0, c1, c2, d0 = self.illu_consts

    Y0 = torch.ones(n_v_full).float() * a0 * c0
    if gamma.is_cuda: Y0 = Y0.cuda()
    norm = norm.view(-1, 3)
    nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
    arrH = []

    arrH.append(Y0)
    arrH.append(-a1 * c1 * ny)
    arrH.append(a1 * c1 * nz)
    arrH.append(-a1 * c1 * nx)
    arrH.append(a2 * c2 * nx * ny)
    arrH.append(-a2 * c2 * ny * nz)
    arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
    arrH.append(-a2 * c2 * nx * nz)
    arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

    H = torch.stack(arrH, 1)
    Y = H.view(n_b, num_vertex, 9)

    # Y shape:[batch,N,9].

    # shape:[batch,N,3]
    lighting = Y.bmm(gamma)

    face_color = face_texture * lighting
    # lighting *= 128

    # print( face_color[0, 5] )
    return face_color, lighting

  def set_Illu_consts(self):
    import numpy as np
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    d0 = 0.5 / np.sqrt(3.0)

    self.illu_consts = [a0, a1, a2, c0, c1, c2, d0]

def get_rotation_matrix(tx, ty, tz):
  m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
  m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
  m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

  m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
  m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
  m_x[:, 0, 0] = 1

  m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
  m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
  m_y[:, 1, 1] = 1

  m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
  m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
  m_z[:, 2, 2] = 1
  return torch.matmul(m_z, torch.matmul(m_y, m_x))

def get_transform_matrices(view):
  b = view.size(0)
  if view.size(1) == 6:
    rx = view[:, 0]
    ry = view[:, 1]
    rz = view[:, 2]
    trans_xyz = view[:, 3:].reshape(b, 1, 3)
  elif view.size(1) == 5:
    rx = view[:, 0]
    ry = view[:, 1]
    rz = view[:, 2]
    delta_xy = view[:, 3:].reshape(b, 1, 2)
    trans_xyz = torch.cat([delta_xy, torch.zeros(b, 1, 1).to(view.device)], 2)
  elif view.size(1) == 3:
    rx = view[:, 0]
    ry = view[:, 1]
    rz = view[:, 2]
    trans_xyz = torch.zeros(b, 1, 3).to(view.device)
  rot_mat = get_rotation_matrix(rx, ry, rz)
  return rot_mat, trans_xyz