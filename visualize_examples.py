"""
Visualize sample predictions for the 250-500px and 500-771px smoke size buckets.
Shows: Original | GT Mask Overlay | Prediction Overlay | Diff Map
"""
import os, sys, tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

ROOT = Path(__file__).resolve().parent / 'FoSp-main'
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

CHECKPOINT = Path(r'C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\epoch_9\best.pth')
IMG_DIR    = Path(r'C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\dataset\SmokeSeg\SmokeSeg\JPEGImages')
MASK_DIR   = Path(r'C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\dataset\SmokeSeg\SmokeSeg\SegmentationClass')
OUT_DIR    = Path('./smoke_test_results')
OUT_DIR.mkdir(exist_ok=True)

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,  57.375],  dtype=np.float32)

# ── Samples to visualize ───────────────────────────────────────────────────
SAMPLES = [
    # (bucket_label, stem, smoke_px, iou, recall, f1)
    ('250–500 px | IoU cao (0.74)',  '20180517_FIRE_smer-tcs10-mobo-c__1526595059_+01260', 457,  0.7379, 0.8315, 0.8492),
    ('250–500 px | IoU trung (0.60)', '20180614_Hope_wc-e-mobo-c__1529001260_+00300',       430,  0.6014, 0.6140, 0.7511),
    ('250–500 px | IoU thấp (0.17)', '20180614_Hope_wc-e-mobo-c__1529001200_+00240',        402,  0.1741, 0.1741, 0.2966),
    ('250–500 px | IoU = 0 (miss)',  '20180726_FIRE_so-w-mobo-c__1532624183_+00780',        250,  0.0000, 0.0000, 0.0000),
    ('500–771 px | IoU cao (0.73)',  '20190924_FIRE_wc-e-mobo-c__1569364103_+01743',        575,  0.7323, 0.9704, 0.8455),
    ('500–771 px | IoU khá  (0.72)', '20190814_FIRE-pi-s-mobo-c__1565823008_+00361',        770,  0.7234, 0.8117, 0.8395),
    ('500–771 px | IoU thấp (0.39)', '20200521_VEGMGMT_ml-w-mobo-c__1590082637_+01980',    720,  0.3917, 0.3917, 0.5629),
    ('500–771 px | IoU = 0 (miss)',  '20180504_FIRE_bh-n-mobo-c__1525471330_+00900',        501,  0.0000, 0.0000, 0.0000),
]


def build_model():
    from mmcv import Config
    from mmseg.models import build_segmentor

    ckpt      = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state     = ckpt['state_dict']
    contract  = ckpt.get('contract', {})
    threshold = float(contract.get('threshold', 0.4))
    size      = int(contract.get('size', 512))

    cfg = Config.fromfile(str(ROOT / 'config/fosp_config.py'))
    cfg.model.pretrained = None
    cfg.model.decode_head.mat_impl = 'ref'
    cfg.model.decode_head.threshold = threshold

    prefix    = 'decode_head.G.generator.'
    mat_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    with tempfile.TemporaryDirectory(prefix='mat_init_') as tmp:
        mat_path = Path(tmp) / 'mat.pth'
        torch.save(mat_state, mat_path)
        cfg.model.decode_head.mat_checkpoint = str(mat_path)
        model = build_segmentor(cfg.model)

    model.load_state_dict(state, strict=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, threshold, size, device


def predict(model, img_path, size, device):
    with Image.open(img_path) as img:
        rgb = img.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    arr = (np.array(rgb, dtype=np.float32) - MEAN) / STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model.encode_decode(tensor, [{}])
        prob   = logits.sigmoid()[0, 0].cpu().numpy()
    return prob  # H×W float in [0,1]


def make_overlay(rgb_np, mask_bool, color, alpha=0.50):
    out = rgb_np.copy().astype(float)
    c   = np.array(color, dtype=float)
    out[mask_bool] = out[mask_bool] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def zoom_region(arr, mask_bool, pad=30):
    coords = np.argwhere(mask_bool)
    if len(coords) == 0:
        h, w = arr.shape[:2]
        return arr[h//4:3*h//4, w//4:3*w//4]
    y1, x1 = coords.min(0)
    y2, x2 = coords.max(0)
    h, w   = arr.shape[:2]
    y1 = max(0, y1 - pad); y2 = min(h, y2 + pad)
    x1 = max(0, x1 - pad); x2 = min(w, x2 + pad)
    return arr[y1:y2, x1:x2]


def main():
    print('[*] Building model …', flush=True)
    model, threshold, size, device = build_model()
    print(f'[+] Model ready on {device}', flush=True)

    n = len(SAMPLES)
    fig, axes = plt.subplots(n, 5, figsize=(22, 4.2 * n), dpi=130)
    fig.suptitle(
        'FoSp — Ví dụ dự đoán theo dải kích thước khói\n'
        'Cột: [1] Ảnh gốc  [2] GT Overlay (xanh)  [3] Pred Overlay (đỏ)  '
        '[4] Zoom GT  [5] Zoom Pred',
        fontsize=13, fontweight='bold', y=0.995
    )

    col_titles = ['Ảnh gốc', 'GT Mask\n(smoke thực tế)', 'Pred Mask\n(model dự đoán)',
                  'Zoom GT', 'Zoom Pred']
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11, fontweight='bold', pad=8)

    for row, (label, stem, smoke_px, iou, recall, f1) in enumerate(SAMPLES):
        print(f'  [{row+1}/{n}] {stem[:50]} …', flush=True)

        # --- find files ---
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            p = IMG_DIR / (stem + ext)
            if p.exists():
                img_path = p; break
        mask_path = MASK_DIR / (stem + '.png')

        if img_path is None:
            for ax in axes[row]:
                ax.axis('off')
            axes[row, 0].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            continue

        # --- load ---
        with Image.open(img_path) as im:
            orig_rgb = np.array(im.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR))
        gt_arr  = np.array(Image.open(mask_path).resize((512, 512), Image.Resampling.NEAREST))
        gt_mask = gt_arr > 0

        # --- predict ---
        prob      = predict(model, img_path, size, device)
        # resize prob to 512 for display
        prob_disp = F.interpolate(
            torch.from_numpy(prob).unsqueeze(0).unsqueeze(0),
            size=(512, 512), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        pred_mask = prob_disp > threshold

        # --- overlays ---
        gt_overlay   = make_overlay(orig_rgb, gt_mask,   [30, 220, 60],  alpha=0.55)   # green
        pred_overlay = make_overlay(orig_rgb, pred_mask, [255, 50, 50],  alpha=0.55)   # red

        # --- zoom around GT region ---
        zoom_gt   = zoom_region(gt_overlay,   gt_mask,   pad=35)
        zoom_pred = zoom_region(pred_overlay, gt_mask,   pad=35)  # zoom same region

        row_label = (f'{label}\n'
                     f'smoke={smoke_px} px  |  IoU={iou:.4f}  Recall={recall:.4f}  F1={f1:.4f}')

        for ci, (img_arr, cmap) in enumerate([
            (orig_rgb,    None),
            (gt_overlay,  None),
            (pred_overlay, None),
            (zoom_gt,     None),
            (zoom_pred,   None),
        ]):
            ax = axes[row, ci]
            ax.imshow(img_arr, cmap=cmap)
            ax.axis('off')
            if ci == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold',
                               rotation=0, labelpad=160, va='center', ha='right')

        # Divider line between buckets
        if row == 3:
            for ci in range(5):
                axes[row, ci].spines['bottom'].set_visible(True)
                axes[row, ci].spines['bottom'].set_linewidth(2)
                axes[row, ci].spines['bottom'].set_color('#e63946')

    # Legend
    legend_elements = [
        mpatches.Patch(color=(30/255, 220/255, 60/255), label='Ground Truth smoke (GT)'),
        mpatches.Patch(color=(255/255, 50/255, 50/255), label='Model prediction (Pred)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.005))

    out_path = OUT_DIR / 'examples_250_771px.png'
    plt.tight_layout(rect=[0, 0.015, 1, 0.995])
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'\n[+] Đã lưu ảnh ví dụ tại: {out_path.resolve()}')


if __name__ == '__main__':
    main()
