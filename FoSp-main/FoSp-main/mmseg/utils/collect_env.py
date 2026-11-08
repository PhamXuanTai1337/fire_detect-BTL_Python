# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmseg
import platform
import warnings

import mmcv
import torch


def collect_env():
    """Collect the information of the running environments."""
    try:
        env_info = collect_base_env()
    except Exception as exc:
        # Compiler discovery (e.g. no MSVC) must not block reference-op training.
        warnings.warn(f'Incomplete environment report: {exc}')
        env_info = dict(Python=platform.python_version(),
                        PyTorch=str(torch.__version__), MMCV=mmcv.__version__,
                        CUDA=torch.version.cuda)
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
