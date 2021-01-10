from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import cv2
from glob import glob
import os
import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def get_frames(video_name):
  if not video_name:
    cap = cv2.VideoCapture(0)
    for i in range(5):
      cap.read()
    while True:
      ret, frame = cap.read()
      if ret:
        yield frame
      else:
        break
  elif video_name.endswith('avi') or video_name.endswith('mp4'):
    cap = cv2.VideoCapture(video_name)
    while True:
      ret, frame = cap.read()
      if ret:
        yield frame
      else:
        break
  else:
    images = glob(os.path.join(video_name, '*.jpg'))
    images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
      frame = cv2.imread(img)
      yield frame

mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

def preprocess(image, res):
  h, w, _ = image.shape
  padh, padw = max(h, w) - h, max(h, w) - w
  image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')
  image = cv2.resize(image, (res, res))
  image = (image.astype(np.float32) / 255. - mean) / std
  meta = {'s': max(h, w) / res}
  return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda(), meta

def construct_meshes(shape, texture, face):
  nV = shape.size(2)
  Verts, Faces, Textures = [], [], []
  for i in range(len(shape)):
    V_ = shape[i]
    T_ = texture[i]

    range_ = torch.arange(V_.size(0)).view(-1, 1) * nV
    F_ = face.expand(V_.size(0), -1) + range_.cuda()
    # F_ = F_.view(-1, 3).float()
    Verts.append(V_.reshape(-1, 3))
    Textures.append(T_.reshape(-1, 3))
    Faces.append(F_.reshape(-1, 3).float())

  meshes = Meshes(verts=Verts, faces=Faces,
                  textures=TexturesVertex(verts_features=Textures))
  return meshes