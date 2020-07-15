from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data.dataloader import _use_shared_memory,numpy_type_map,int_classes,string_classes
from torch._six import container_abcs, string_classes, int_classes
import collections
import re

from models.networks.MABN import MABN2d

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.eval()
        # module.weight.requires_grad = False
        # module.bias.requires_grad   = False

def use_mabn(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == nn.BatchNorm2d:
            print('replaced: ', name, attr_str)
            setattr(m, attr_str, MABN2d(target_attr.num_features, eps=target_attr.eps))
    for n, ch in m.named_children():
        use_mabn(ch, n)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        res = {key: collate([d[key] for d in batch]) for key in batch[0] if key!='instance_mask'}
        if 'instance_mask' in batch[0]:
            max_obj = max([d['instance_mask'].shape[0] for d in batch])
            instance_mask = torch.zeros(len(batch),max_obj,*(batch[0]['instance_mask'].shape[1:]))
            for i in range(len(batch)):
                num_obj = batch[i]['instance_mask'].shape[0]
                instance_mask[i,:num_obj] = torch.as_tensor(batch[i]['instance_mask'])
            res.update({'instance_mask':instance_mask})
        return res
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(elem_type))

# def collate(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""

#     error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
#     elem_type = type(batch[0])
#     if isinstance(batch[0], torch.Tensor):
#         out = None
#         if _use_shared_memory:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = batch[0].storage()._new_shared(numel)
#             out = batch[0].new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         elem = batch[0]
#         if elem_type.__name__ == 'ndarray':
#             # array of string classes and object
#             if re.search('[SaUO]', elem.dtype.str) is not None:
#                 raise TypeError(error_msg.format(elem.dtype))

#             return torch.stack([torch.from_numpy(b) for b in batch], 0)
#         if elem.shape == ():  # scalars
#             py_type = float if elem.dtype.name.startswith('float') else int
#             return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
#     elif isinstance(batch[0], int_classes):
#         return torch.LongTensor(batch)
#     elif isinstance(batch[0], float):
#         return torch.DoubleTensor(batch)
#     elif isinstance(batch[0], string_classes):
#         return batch
#     elif isinstance(batch[0], collections.Mapping):
#         res =  {key: collate([d[key] for d in batch]) for key in batch[0] if key!='instance_mask'}
#         if 'instance_mask' in batch[0]:
#             max_obj = max([d['instance_mask'].shape[0] for d in batch])
#             instance_mask = torch.zeros(len(batch),max_obj,*(batch[0]['instance_mask'].shape[1:]))
#             for i in range(len(batch)):
#                 num_obj = batch[i]['instance_mask'].shape[0]
#                 instance_mask[i,:num_obj] = torch.from_numpy(batch[i]['instance_mask'])
#             res.update({'instance_mask':instance_mask})
#         return res
#     elif isinstance(batch[0], collections.Sequence):
#         transposed = zip(*batch)
#         return [collate(samples) for samples in transposed]

#     raise TypeError((error_msg.format(type(batch[0]))))