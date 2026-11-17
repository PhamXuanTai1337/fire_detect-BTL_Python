"""Aspect-preserving 512-square working image with bottom/right padding."""
import cv2
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class SmokeLetterbox:
    def __init__(self, size=512, image_pad_value=0, ignore_index=255):
        if size != 512:
            raise ValueError('FoSp nine-crop geometry requires size=512.')
        self.size, self.image_pad_value, self.ignore_index = size, image_pad_value, ignore_index

    def __call__(self, results):
        image = results['img']
        h, w = image.shape[:2]
        ratio = self.size / max(h, w)
        rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
        resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((512, 512, 3), self.image_pad_value, dtype=image.dtype)
        padded[:rh, :rw] = resized
        results['img'] = padded
        # img_shape excludes padding, matching the existing inference metadata.
        results['img_shape'], results['pad_shape'] = resized.shape, padded.shape
        results['scale_factor'] = np.array([rw / w, rh / h, rw / w, rh / h], dtype=np.float32)
        results['keep_ratio'] = True
        for key in results.get('seg_fields', []):
            mask = cv2.resize(results[key], (rw, rh), interpolation=cv2.INTER_NEAREST)
            target = np.full((512, 512), self.ignore_index, dtype=mask.dtype)
            target[:rh, :rw] = mask
            results[key] = target
        return results

    def __repr__(self):
        return 'SmokeLetterbox(size=512, padding=bottom_right)'
