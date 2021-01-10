from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2
  if kernel == 2:
    hm_pad = F.pad(heat, [0, 1, 0, 1])
    hmax = F.max_pool2d(hm_pad, (kernel, kernel), stride=1, padding=pad)
  else:
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk(scores, K):
  b, c, h, w = scores.size()
  assert c == 1
  topk_scores, topk_inds = torch.topk(scores.view(b, -1), K)

  topk_inds = topk_inds % (h * w)
  topk_ys = (topk_inds / w).int().float()
  topk_xs = (topk_inds % w).int().float()
  return topk_scores, topk_inds, topk_ys, topk_xs

def decode(image, model):
    output = model(image)[0]
    hm = output['hm']
    hm.sigmoid_()
    score = 0.5
    hm = _nms(hm, 5)
    K = int((hm > score).float().sum())
    topk_scores, topk_inds, topk_ys, topk_xs = _topk(hm, K)
    return [output, topk_scores, topk_inds, topk_ys, topk_xs]