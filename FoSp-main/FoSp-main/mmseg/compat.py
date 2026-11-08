"""Small compatibility fixes for the vendored MMCV 1.x training stack."""
import inspect

import torch
from mmcv.parallel import _functions
from torch.nn.parallel._functions import _get_stream


def apply_compat():
    # New PyTorch takes torch.device; MMCV 1.x still passes an integer GPU id.
    annotation = inspect.signature(_get_stream).parameters['device'].annotation
    if annotation in (torch.device, 'torch.device'):
        def get_stream(device):
            return _get_stream(torch.device('cuda', device)
                               if isinstance(device, int) else device)
        _functions._get_stream = get_stream
