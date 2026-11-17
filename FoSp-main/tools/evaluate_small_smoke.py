"""Evaluate a fixed checkpoint/threshold on disjoint val/test image IDs.

Example: python tools/evaluate_small_smoke.py config/fosp_small_smoke.py
  --checkpoint PATH --split val --mode selective --output-dir PATH
Use baseline config for the original baseline; crop config supports whole/all/selective.
No threshold search, checkpoint selection or training is performed here.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from mmcv import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmseg.smoke_inference import load_for_inference, prepare_image, restore_count_map
from mmseg.smoke_metrics import independent_splits, measure_image, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--mode', choices=['whole', 'all', 'selective'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, help='Optional execution smoke test; reported as a subset.')
    parser.add_argument('--save-debug', action='store_true', help='Save coverage maps in original coordinates.')
    args = parser.parse_args()
    cfg = Config.fromfile(str(args.config))
    splits = independent_splits(cfg.data)
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive.')
    if args.mode:
        if cfg.model.type == 'SelectiveCropFOSP':
            cfg.model.crop_cfg.mode = args.mode
        elif args.mode != 'whole':
            parser.error('Original baseline only supports --mode whole.')
    model, cfg = load_for_inference(cfg, args.checkpoint, args.device)
    names = splits[args.split][:args.limit]
    spec = cfg.data[args.split]
    root = Path(spec.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'summary.json').exists():
        raise ValueError('Choose a fresh output directory to preserve previous evaluations.')
    rows = []
    for name in names:
        stem = Path(name).stem
        image_path = root / spec.img_dir / (stem + '.jpg')
        mask_path = root / spec.ann_dir / (stem + '.png')
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        image, meta = prepare_image(cfg, image_path, args.device, split=args.split)
        with torch.no_grad():
            probability = model.inference(image, meta, rescale=True)[0, 0].cpu().numpy()
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # GT is loaded only after prediction, and never passed to inference.
        mask = np.array(Image.open(mask_path))
        debug = getattr(model, 'last_crop_debug', [])
        if debug:
            item = debug[0]
            counts = restore_count_map(item, meta[0]).cpu().numpy()
            indices, fallback = item['selected'], item['fallback']
            scores = None if item['scores'] is None else item['scores'].cpu().tolist()
        else:
            counts = np.ones_like(probability)
            indices, fallback, scores = [], False, None
        row = dict(stem=stem, **measure_image(probability, mask, counts > 0, model.decode_head.threshold),
                   crop_count=len(indices), full_image_passes=int(not bool(debug)),
                   fallback=fallback, latency_seconds=elapsed,
                   selected=json.dumps(indices), scores=json.dumps(scores))
        rows.append(row)
        if args.save_debug:
            np.save(args.output_dir / f'{stem}_count_map.npy', counts)
            Image.fromarray((counts == 0).astype(np.uint8) * 255).save(args.output_dir / f'{stem}_uncovered.png')
        print(f'[{len(rows)}/{len(names)}] {stem}: IoU={row["foreground_iou"]:.4f}', flush=True)
    summary = dict(checkpoint=str(args.checkpoint.resolve()), config=str(args.config.resolve()),
        split=args.split, independent_image_ids=True, subset=args.limit is not None,
        mode=cfg.model.get('crop_cfg', {}).get('mode', 'whole'),
        segmentation_threshold=model.decode_head.threshold,
        crop_threshold=cfg.model.get('crop_cfg', {}).get('threshold'),
        latency_scope='image loading + preprocessing + selector + FoSp + merging + restoration + CPU transfer; includes cold first image',
        size_basis='foreground pixel count in original GT mask', **summarize(rows))
    with (args.output_dir / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
