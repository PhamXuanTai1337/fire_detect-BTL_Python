"""
Test FoSp best.pth on SmokeSeg images with smoke pixel count < threshold (default P50=771).
Metrics: F1, Recall, Foreground IoU  (Global-pixel aggregation + per-image mean).

Usage:
  .venv-mat\\Scripts\\python.exe test_small_smoke.py
  .venv-mat\\Scripts\\python.exe test_small_smoke.py --threshold 500 --max_images 300
"""
import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── 1. Env & Paths ─────────────────────────────────────────────────────────
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

ROOT = Path(__file__).resolve().parent / 'FoSp-main'
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CHECKPOINT = Path(r'C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\epoch_9\best.pth')
IMG_DIR    = Path(r'C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\dataset\SmokeSeg\SmokeSeg\JPEGImages')
MASK_DIR   = Path(r'C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\dataset\SmokeSeg\SmokeSeg\SegmentationClass')

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,  57.375],  dtype=np.float32)


# ── 2. Build Model ──────────────────────────────────────────────────────────
def build_model(checkpoint_path: Path, device: str):
    from mmcv import Config
    from mmseg.models import build_segmentor

    ckpt      = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state     = ckpt['state_dict']
    contract  = ckpt.get('contract', {})
    threshold = float(contract.get('threshold', 0.4))
    size      = int(contract.get('size', 512))

    cfg = Config.fromfile(str(ROOT / 'config/fosp_config.py'))
    cfg.model.pretrained = None
    cfg.model.decode_head.mat_impl = 'ref'
    cfg.model.decode_head.threshold = threshold

    prefix   = 'decode_head.G.generator.'
    mat_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not mat_state:
        raise ValueError('Checkpoint has no embedded MAT weights.')

    with tempfile.TemporaryDirectory(prefix='mat_init_') as tmp:
        mat_path = Path(tmp) / 'mat.pth'
        torch.save(mat_state, mat_path)
        cfg.model.decode_head.mat_checkpoint = str(mat_path)
        model = build_segmentor(cfg.model)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f'[+] Model loaded (epoch {ckpt.get("epoch")}, threshold={threshold})', flush=True)
    return model, threshold, size


# ── 3. Select images ───────────────────────────────────────────────────────
def select_images(smoke_px_threshold: int, max_images: int, seed: int = 42):
    """Return list of stems whose GT mask has < smoke_px_threshold smoke pixels."""

    def count_smoke(mask_file: Path) -> int:
        arr = np.array(Image.open(mask_file))
        return int(np.count_nonzero(arr > 0))

    mask_files = [f for f in MASK_DIR.iterdir()
                  if f.suffix.lower() == '.png' and not f.name.startswith('.')]

    print(f'[*] Scanning {len(mask_files)} masks …', flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        counts = list(ex.map(count_smoke, mask_files))

    # Filter by smoke size
    pairs = [(f.stem, c) for f, c in zip(mask_files, counts) if 0 < c < smoke_px_threshold]
    pairs.sort(key=lambda x: x[1])                # sort ascending

    if len(pairs) > max_images:
        # Sample evenly across the range so we cover the whole distribution
        indices = np.round(np.linspace(0, len(pairs) - 1, max_images)).astype(int)
        pairs = [pairs[i] for i in indices]

    print(f'[+] Selected {len(pairs)} images (smoke < {smoke_px_threshold} px, '
          f'range {pairs[0][1]}–{pairs[-1][1]} px)', flush=True)
    return [p[0] for p in pairs]


# ── 4. Preprocess ──────────────────────────────────────────────────────────
def preprocess(img_path: Path, size: int) -> torch.Tensor:
    with Image.open(img_path) as img:
        rgb = img.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    arr = (np.array(rgb, dtype=np.float32) - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)


# ── 5. Per-image metrics ───────────────────────────────────────────────────
def image_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt: boolean 2-D arrays of same shape.
    Returns dict with tp, fp, fn, foreground_iou, recall, f1.
    """
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    iou    = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    recall = tp / (tp + fn)      if (tp + fn) > 0      else 0.0
    prec   = tp / (tp + fp)      if (tp + fp) > 0      else 0.0
    f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn,
                foreground_iou=iou, recall=recall, precision=prec, f1=f1)


# ── 6. Main evaluation loop ────────────────────────────────────────────────
def evaluate(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[*] Device: {device}', flush=True)

    stems = select_images(args.smoke_px_threshold, args.max_images, args.seed)

    model, threshold, size = build_model(CHECKPOINT, device)

    per_image_results = []
    total_tp = total_fp = total_fn = 0
    t_start = time.perf_counter()

    for idx, stem in enumerate(stems, 1):
        # --- find files ---
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            p = IMG_DIR / (stem + ext)
            if p.exists():
                img_path = p
                break
        mask_path = MASK_DIR / (stem + '.png')

        if img_path is None or not mask_path.exists():
            print(f'  [!] Skip {stem}: missing image or mask', flush=True)
            continue

        # --- load GT mask ---
        gt_arr  = np.array(Image.open(mask_path))
        gt_mask = gt_arr > 0  # boolean H×W

        # --- inference ---
        tensor = preprocess(img_path, size).to(device)
        with torch.no_grad():
            logits = model.encode_decode(tensor, [{}])
            prob   = F.interpolate(logits, size=gt_arr.shape[:2],
                                   mode='bilinear', align_corners=False)
            prob   = prob.sigmoid()[0, 0].cpu().numpy()

        pred_mask = prob > threshold  # boolean H×W

        # --- metrics ---
        m = image_metrics(pred_mask, gt_mask)
        m['stem']      = stem
        m['smoke_px']  = int(gt_mask.sum())
        per_image_results.append(m)

        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']

        if idx % 50 == 0 or idx == len(stems):
            elapsed = time.perf_counter() - t_start
            print(f'  [{idx:4d}/{len(stems)}] {stem[:40]:40s} '
                  f'| smoke={m["smoke_px"]:5d} px '
                  f'| IoU={m["foreground_iou"]:.4f} '
                  f'| Recall={m["recall"]:.4f} '
                  f'| F1={m["f1"]:.4f} '
                  f'| {elapsed:.1f}s elapsed', flush=True)

    if not per_image_results:
        print('[!] No images processed.', flush=True)
        return

    elapsed_total = time.perf_counter() - t_start

    # ── Global-pixel metrics ──────────────────────────────────────────────
    g_iou    = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    g_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    g_prec   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    g_f1     = (2 * g_prec * g_recall / (g_prec + g_recall)) if (g_prec + g_recall) > 0 else 0.0

    # ── Per-image mean metrics ────────────────────────────────────────────
    m_iou    = np.mean([r['foreground_iou'] for r in per_image_results])
    m_recall = np.mean([r['recall']         for r in per_image_results])
    m_f1     = np.mean([r['f1']             for r in per_image_results])

    # ── Breakdown by smoke-size buckets ──────────────────────────────────
    buckets = [(0, 100), (100, 250), (250, 500), (500, args.smoke_px_threshold)]

    print('\n' + '=' * 72)
    print('  ĐÁNH GIÁ MÔ HÌNH FoSp — TẬP ẢNH KHÓI NHỎ (< {} px)'.format(args.smoke_px_threshold))
    print('=' * 72)
    print(f'  Checkpoint : {CHECKPOINT}')
    print(f'  Threshold  : {threshold}  |  Input size: {size}×{size}')
    print(f'  Số ảnh test: {len(per_image_results)}  '
          f'(smoke range {min(r["smoke_px"] for r in per_image_results)}'
          f' – {max(r["smoke_px"] for r in per_image_results)} px)')
    print(f'  Thời gian  : {elapsed_total:.1f}s '
          f'({elapsed_total / len(per_image_results):.2f}s/ảnh)\n')

    print('  ── GLOBAL-PIXEL (gộp toàn bộ pixel) ──')
    print(f'  Foreground IoU : {g_iou:.4f}')
    print(f'  Recall         : {g_recall:.4f}')
    print(f'  F1             : {g_f1:.4f}')
    print(f'  Precision      : {g_prec:.4f}')
    print(f'  TP / FP / FN   : {total_tp:,} / {total_fp:,} / {total_fn:,}\n')

    print('  ── PER-IMAGE MEAN (trung bình từng ảnh) ──')
    print(f'  Foreground IoU : {m_iou:.4f}')
    print(f'  Recall         : {m_recall:.4f}')
    print(f'  F1             : {m_f1:.4f}\n')

    print('  ── PHÂN TÍCH THEO DẢI KÍCH THƯỚC KHÓI ──')
    print(f'  {"Dải pixel":<18} {"Số ảnh":>7}  {"IoU":>7}  {"Recall":>8}  {"F1":>7}')
    print('  ' + '-' * 55)
    for lo, hi in buckets:
        sub = [r for r in per_image_results if lo <= r['smoke_px'] < hi]
        if not sub:
            continue
        bi   = np.mean([r['foreground_iou'] for r in sub])
        br   = np.mean([r['recall']         for r in sub])
        bf   = np.mean([r['f1']             for r in sub])
        print(f'  [{lo:4d} – {hi:4d}) px  {len(sub):>7}  {bi:>7.4f}  {br:>8.4f}  {bf:>7.4f}')
    print('=' * 72 + '\n')

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir = Path('./smoke_test_results')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f'results_small_smoke_under_{args.smoke_px_threshold}px.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('stem,smoke_px,foreground_iou,recall,precision,f1,tp,fp,fn\n')
        for r in per_image_results:
            f.write(f"{r['stem']},{r['smoke_px']},"
                    f"{r['foreground_iou']:.6f},{r['recall']:.6f},"
                    f"{r['precision']:.6f},{r['f1']:.6f},"
                    f"{r['tp']},{r['fp']},{r['fn']}\n")
    print(f'[+] Chi tiết từng ảnh đã lưu tại: {csv_path.resolve()}')

    # ── Summary JSON ──────────────────────────────────────────────────────
    import json
    summary = dict(
        checkpoint=str(CHECKPOINT), threshold=threshold,
        smoke_px_threshold=args.smoke_px_threshold, n_images=len(per_image_results),
        global_pixel=dict(foreground_iou=round(g_iou, 6), recall=round(g_recall, 6),
                          f1=round(g_f1, 6), precision=round(g_prec, 6),
                          tp=total_tp, fp=total_fp, fn=total_fn),
        per_image_mean=dict(foreground_iou=round(float(m_iou), 6),
                            recall=round(float(m_recall), 6),
                            f1=round(float(m_f1), 6))
    )
    json_path = out_dir / f'summary_small_smoke_under_{args.smoke_px_threshold}px.json'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[+] Tóm tắt kết quả đã lưu tại: {json_path.resolve()}')


# ── 7. Entry point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke_px_threshold', type=int, default=771,
                        help='Chỉ test ảnh có số pixel khói DƯỚI ngưỡng này (mặc định: P50=771)')
    parser.add_argument('--max_images', type=int, default=600,
                        help='Số lượng ảnh tối đa để test (mặc định: 600)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
