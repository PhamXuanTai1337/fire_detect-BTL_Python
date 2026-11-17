"""Foreground metrics, selector coverage and independent split checks."""
from pathlib import Path

import cv2
import numpy as np


def independent_splits(data):
    splits = {}
    for name in ('train', 'val', 'test'):
        spec = data[name]
        path = Path(spec['split'])
        if not path.is_absolute():
            path = Path(spec['data_root']) / path
        names = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines()
                 if line.strip()]
        stems = [Path(value).stem for value in names]
        if not names or len(stems) != len(set(stems)):
            raise ValueError(f'Empty split or duplicate image IDs: {path}')
        splits[name] = names
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = {Path(x).stem for x in splits[a]} & {Path(x).stem for x in splits[b]}
        if overlap:
            raise ValueError(f'Splits {a}/{b} overlap: {sorted(overlap)[:5]}')
    return splits


def foreground_metrics(tp, fp, fn):
    def ratio(a, b):
        return float(a / b) if b else 0.0
    return dict(foreground_iou=ratio(tp, tp + fp + fn), recall=ratio(tp, tp + fn),
                precision=ratio(tp, tp + fp), f1=ratio(2 * tp, 2 * tp + fp + fn))


def measure_image(probability, mask, covered, threshold=0.4):
    if probability.shape != mask.shape or covered.shape != mask.shape:
        raise ValueError('Probability, coverage and GT must share original coordinates.')
    if not np.isfinite(probability).all():
        raise ValueError('Non-finite prediction.')
    valid = mask != 255
    if np.any(valid & (mask != 0) & (mask != 1)):
        raise ValueError('Expected mask labels 0, 1, 255.')
    smoke = mask == 1
    pred = (probability > threshold) & valid
    tp, fp, fn = int((pred & smoke).sum()), int((pred & ~smoke).sum()), int((~pred & smoke).sum())
    components, labels = cv2.connectedComponents(smoke.astype(np.uint8), connectivity=8)
    regions_hit = regions_full = 0
    for label in range(1, components):
        pixels = covered[labels == label]
        regions_hit += int(pixels.any())
        regions_full += int(pixels.all())
    return dict(smoke_px=int(smoke.sum()), valid_px=int(valid.sum()),
                background_px=int((valid & ~smoke).sum()), tp=tp, fp=fp, fn=fn,
                predicted_px=int(pred.sum()), empty_prediction=not bool(pred.any()),
                no_tp=tp == 0, covered_smoke_px=int((smoke & covered).sum()),
                smoke_regions=components - 1, covered_regions=regions_hit,
                fully_covered_regions=regions_full, **foreground_metrics(tp, fp, fn))


def summarize(rows):
    valid_rows = [r for r in rows if r['valid_px'] > 0]
    positive = [r for r in valid_rows if r['smoke_px'] > 0]
    negative = [r for r in valid_rows if r['smoke_px'] == 0]
    totals = {k: sum(r[k] for r in valid_rows) for k in
              ('tp', 'fp', 'fn', 'covered_smoke_px', 'smoke_px', 'covered_regions',
               'fully_covered_regions', 'smoke_regions')}
    def mean(items, key):
        return float(np.mean([r[key] for r in items])) if items else None
    def ratio(a, b):
        return float(a / b) if b else None
    keys = ('foreground_iou', 'recall', 'precision', 'f1')
    buckets = {}
    for lo, hi in ((1, 100), (100, 250), (250, 500), (500, 771), (771, 1024), (1024, float('inf'))):
        items = [r for r in positive if lo <= r['smoke_px'] < hi]
        buckets[f'[{lo},{hi})'] = dict(n_images=len(items),
            **{key: mean(items, key) for key in keys},
            empty_prediction_rate=mean(items, 'empty_prediction'), no_tp_rate=mean(items, 'no_tp'))
    return dict(n_images=len(rows), n_valid_images=len(valid_rows), n_positive=len(positive),
        n_negative=len(negative), all_ignore_images=len(rows) - len(valid_rows),
        global_pixel=dict(**{k: totals[k] for k in ('tp', 'fp', 'fn')},
                          **foreground_metrics(totals['tp'], totals['fp'], totals['fn'])),
        per_image_positive_mean={key: mean(positive, key) for key in keys},
        per_image_valid_mean={key: mean(valid_rows, key) for key in keys},
        zero_denominator_metric_value=0.0,
        positive_empty_prediction_rate=mean(positive, 'empty_prediction'),
        positive_no_tp_rate=mean(positive, 'no_tp'),
        negative_false_alarm_rate=ratio(sum(r['fp'] > 0 for r in negative), len(negative)),
        negative_pixel_fp_rate=ratio(sum(r['fp'] for r in negative), sum(r['background_px'] for r in negative)),
        smoke_pixel_coverage=ratio(totals['covered_smoke_px'], totals['smoke_px']),
        region_any_pixel_coverage=ratio(totals['covered_regions'], totals['smoke_regions']),
        region_full_coverage=ratio(totals['fully_covered_regions'], totals['smoke_regions']),
        mean_crop_count=mean(rows, 'crop_count'), fallback_rate=mean(rows, 'fallback'),
        latency_seconds_mean=mean(rows, 'latency_seconds'),
        latency_seconds_p95=float(np.percentile([r['latency_seconds'] for r in rows], 95)) if rows else None,
        size_buckets=buckets)
