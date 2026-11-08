"""Validate all split paths, image/mask sizes, labels and train/val leakage."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from mmcv import Config


def check(config, splits):
    cfg = Config.fromfile(str(config))
    checkpoint = Path(cfg.model.decode_head.mat_checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f'MAT checkpoint missing: {checkpoint}')
    report, identifiers, errors = {}, {}, []
    for split in splits:
        spec = cfg.data[split]
        root = Path(spec.data_root)
        split_file = root / spec.split
        if not split_file.is_file():
            errors.append(f'Missing split: {split_file}')
            continue
        lines = split_file.read_text(encoding='utf-8-sig').splitlines()
        names = [line.strip() for line in lines if line.strip()]
        if len(names) != len(lines):
            errors.append(f'{split}: remove empty lines from {split_file}')
        if not names or len(names) != len(set(names)):
            errors.append(f'{split}: empty split or duplicate IDs')
        identifiers[split] = set(names)
        counts = {0: 0, 1: 0, 255: 0}
        for name in names:
            try:
                with Image.open(root / spec.img_dir / (name + '.jpg')) as image:
                    image.load()
                    size = image.size
                with Image.open(root / spec.ann_dir / (name + '.png')) as mask_image:
                    mask = np.asarray(mask_image)
                if mask.ndim != 2 or mask.shape != (size[1], size[0]):
                    raise ValueError('mask must be single channel/indexed PNG, same size as image')
                values, numbers = np.unique(mask, return_counts=True)
                if not set(values.tolist()).issubset({0, 1, 255}):
                    raise ValueError(f'unsupported labels {values.tolist()}')
                for value, count in zip(values, numbers):
                    counts[int(value)] += int(count)
            except (OSError, ValueError) as exc:
                errors.append(f'{split}/{name}: {exc}')
        if counts[1] == 0:
            errors.append(f'{split}: no label 1 (smoke). Label 255 is ignored, NOT smoke. '
                          'Check mask encoding before training.')
        report[split] = {'images': len(names), 'pixel_counts': counts}
    if 'train' in identifiers:
        for split in ('val', 'test'):
            overlap = identifiers['train'] & identifiers.get(split, set())
            if overlap:
                errors.append(f'train/{split} overlap: {len(overlap)} IDs; e.g. {sorted(overlap)[:3]}')
    print(json.dumps(report, indent=2))
    if errors:
        raise ValueError('\n'.join(errors[:30]) +
                         (f'\n... {len(errors)} total errors' if len(errors) > 30 else ''))
    print('PASS: all selected images, masks, splits and MAT path checked.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='config/fosp_kaggle.py')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    args = parser.parse_args()
    try:
        check(args.config, args.splits)
    except ValueError as exc:
        raise SystemExit(str(exc))
