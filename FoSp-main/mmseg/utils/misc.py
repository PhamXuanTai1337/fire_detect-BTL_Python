# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import re
import warnings


def find_latest_checkpoint(path, suffix='pth'):
    """This function is for finding the latest checkpoint.

    It will be used when automatically resume, modified from
    https://github.com/open-mmlab/mmdetection/blob/dev-v2.20.0/mmdet/utils/misc.py

    Args:
        path (str): The path to find checkpoints.
        suffix (str): File extension for the checkpoint. Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    """
    if not osp.exists(path):
        warnings.warn("The path of the checkpoints doesn't exist.")
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    candidates = []
    for checkpoint in glob.glob(osp.join(path, f'*.{suffix}')):
        match = re.fullmatch(r'(?:iter|epoch)_(\d+)\.' + re.escape(suffix),
                             osp.basename(checkpoint))
        if match:
            candidates.append((int(match.group(1)), checkpoint))
    if not candidates:
        warnings.warn('No periodic checkpoints found in the path')
        return None
    return max(candidates)[1]
