import torch.nn as nn
from pytorch3d.renderer import DirectionalLights, Materials
class Renderer(nn.Module):
  def __init__(self, rasterizer, shader_rgb, shader_mask, cameras):
    super(Renderer, self).__init__()
    self.cameras = cameras
    self.rasterizer = rasterizer
    self.shader_rgb = shader_rgb
    self.shader_mask = shader_mask
    self.lights = DirectionalLights(
            ambient_color=((1, 1, 1),),
            diffuse_color=((0., 0., 0.),),
            specular_color=((0., 0., 0.),),
            direction=((0, 0, 1),),
            device='cuda',
        )
    self.materials = Materials(device='cuda')

  def forward(self, meshes):
    fragments = self.rasterizer(meshes, cameras=self.cameras)
    image = self.shader_rgb(fragments, meshes, cameras=self.cameras,
                            lights=self.lights, materials=self.materials)
    mask = self.shader_mask(fragments, meshes, cameras=self.cameras)
    return image, mask, fragments