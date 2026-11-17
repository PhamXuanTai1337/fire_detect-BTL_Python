"""Luminance-only CLAHE shared by training and image inference."""
import cv2
import numpy as np

from ..builder import PIPELINES


def apply_lab_clahe(image, color_order, clip_limit=3.0, tile_grid_size=(16, 16)):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('LAB CLAHE expects an unnormalized uint8 H x W x 3 image.')
    if color_order not in ('RGB', 'BGR'):
        raise ValueError('color_order must explicitly be RGB or BGR.')
    if clip_limit <= 0 or len(tile_grid_size) != 2 or any(
            not isinstance(x, int) or x <= 0 for x in tile_grid_size):
        raise ValueError('Invalid CLAHE clip limit or grid dimensions.')
    to_lab = cv2.COLOR_RGB2LAB if color_order == 'RGB' else cv2.COLOR_BGR2LAB
    from_lab = cv2.COLOR_LAB2RGB if color_order == 'RGB' else cv2.COLOR_LAB2BGR
    lab = cv2.cvtColor(image, to_lab)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, from_lab)


@PIPELINES.register_module()
class LabCLAHE:
    """Apply after photometric augmentation, before Normalize and nine crops."""
    def __init__(self, clip_limit=3.0, tile_grid_size=(16, 16), color_order='BGR'):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.color_order = color_order

    def __call__(self, results):
        if results.get('clahe_applied', False):
            raise ValueError('CLAHE was already applied to this panorama; do not reapply on crops.')
        # LoadImageFromFile installs identity normalization metadata before any
        # normalization has actually occurred. Accept that sentinel only.
        norm = results.get('img_norm_cfg')
        identity = (norm is not None and 'mean' in norm and 'std' in norm and
                    np.all(np.asarray(norm['mean']) == 0) and
                    np.all(np.asarray(norm['std']) == 1) and not norm.get('to_rgb', False))
        if norm is not None and not identity:
            raise ValueError('LabCLAHE must run before Normalize.')
        results['img'] = apply_lab_clahe(
            results['img'], self.color_order, self.clip_limit, self.tile_grid_size)
        results['clahe_applied'] = True
        return results

    def __repr__(self):
        return (f'LabCLAHE(clip_limit={self.clip_limit}, '
                f'tile_grid_size={self.tile_grid_size}, color_order={self.color_order!r})')
