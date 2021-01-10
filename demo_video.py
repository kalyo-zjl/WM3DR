from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from lib.decode import decode
from lib.model import create_model, load_model
from lib.pt_renderer import PtRender
from lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  construct_meshes,
)


def opts():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--input_res', default=512, type=int)
  parser.add_argument('--arch', default='resnet_50', type=str)
  parser.add_argument('--load_model', default='model/final.pth', type=str)
  parser.add_argument('--BFM', default='BFM/mSEmTFK68etc.chj', type=str)
  parser.add_argument('--video_path', default='video/1.mp4', type=str)
  parser.add_argument('--output', default='result', type=str)

  return parser.parse_args()

def main(opt):

  print('Creating model...')
  # opt.input_res = 256
  render = PtRender(opt).cuda().eval()
  opt.heads = {'hm': 1, 'params': 257}
  model = create_model(opt.arch, opt.heads)

  if opt.load_model != '':
    model = load_model(model, opt.load_model)
  model.cuda().eval()

  if not os.path.exists(opt.output):
    os.makedirs(opt.output)

  with torch.no_grad():
    for i, image in enumerate(get_frames(opt.video_path)):
      h, w, _ = image.shape
      plt.imshow(image[..., ::-1])
      plt.show()

      outfile = os.path.join(opt.output, '{}.jpg'.format(str(i).zfill(8)))

      pre_img, meta = preprocess(image.copy(), opt.input_res)

      output, topk_scores, topk_inds, topk_ys, topk_xs = decode(pre_img, model)
      params = _tranpose_and_gather_feat(output['params'], topk_inds)

      B, C, _ = params.size()
      if C == 0:
        print('no face!')
        cv2.imwrite(outfile, image)
        continue

      # 3DMM formation
      # split coefficients
      id_coeff, ex_coeff, tex_coeff, coeff = render.Split_coeff(params.view(-1, params.size(2)))
      render.set_RotTransLight(coeff, topk_inds.view(-1))

      # reconstruct shape
      canoShape_ = render.Shape_formation(id_coeff, ex_coeff)
      rotShape = render.RotTrans(canoShape_)

      Albedo = render.Texture_formation(tex_coeff)

      Texture, lighting = render.Illumination(Albedo, canoShape_)
      Texture = torch.clamp(Texture, 0, 1)

      rotShape = rotShape.view(B, C, -1, 3)
      Texture = Texture.view(B, C, -1, 3)

      # Pytorch3D render
      meshes = construct_meshes(rotShape, Texture, render.BFM.tri.view(1, -1))

      rendered, gpu_masks, depth = render(meshes) # RGB
      rendered = rendered.squeeze(0).detach().cpu().numpy()
      gpu_masks = gpu_masks.squeeze(0).unsqueeze(-1).cpu().numpy()


      # resize to original image
      image = image.astype(np.float32) / 255.
      rendered = cv2.resize(rendered, (max(h, w), max(h, w)))[:h, :w]
      gpu_masks = cv2.resize(gpu_masks, (max(h, w), max(h, w)), interpolation=cv2.INTER_NEAREST)[:h, :w, np.newaxis]
      image_fuse = image * (1 - gpu_masks) + (0.9 * rendered[..., ::-1] + 0.1 * image) * gpu_masks
      # image_fuse = image * (1 - gpu_masks) + rendered[..., ::-1] * gpu_masks

      cv2.imwrite(outfile, (image_fuse * 255).astype(np.uint8))
      plt.imshow(image_fuse[..., ::-1])
      plt.show()


if __name__ == '__main__':
  opt = opts()
  main(opt)
