from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .resnet import get_resnet

_model_factory = {
  'resnet': get_resnet,
}

def create_model(arch, heads):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads)
  return model

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}'.format(model_path))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, ' \
              'loaded shape{}.'.format(
          k, model_state_dict[k].shape, state_dict[k].shape))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))

  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model


def save_model(path, epoch, model, optimizer=None, buffer_remove=False):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()

  state_dict_ = state_dict.copy()
  if buffer_remove:
    # remove named_buffers
    buffer_name = [x[0] for x in model.named_buffers()]
  for key in state_dict:
      if key in buffer_name:
        del state_dict_[key]

  data = {'epoch': epoch,
          'state_dict': state_dict_}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)