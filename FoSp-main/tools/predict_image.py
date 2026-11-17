"""Predict a smoke mask with a full FoSp checkpoint, using its training contract."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmcv import Config
from mmseg.models import build_segmentor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path,
                        help='Use this config and its exact preprocessing (required for crop checkpoints).')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    contract = checkpoint.get('contract', {})
    size = int(contract.get('size', 512))
    threshold = float(contract.get('threshold', 0.4))
    if size != 512:
        raise ValueError('Full MAT requires the 512-pixel training contract.')
    cfg = Config.fromfile(str(args.config or ROOT / 'config/fosp_config.py'))
    cfg.model.pretrained = None
    cfg.model.decode_head.mat_impl = 'ref'
    cfg.model.decode_head.threshold = threshold

    # Initialize MAT from the exact weights embedded in this checkpoint.
    prefix = 'decode_head.G.generator.'
    mat_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not mat_state:
        raise ValueError('Checkpoint does not contain the complete MAT generator.')
    with tempfile.TemporaryDirectory(prefix='mat_init_', dir=args.output_dir) as temporary:
        mat_path = Path(temporary) / 'mat.pth'
        torch.save(mat_state, mat_path)
        cfg.model.decode_head.mat_checkpoint = str(mat_path)
        model = build_segmentor(cfg.model)
    model.load_state_dict(state, strict=True)
    print(f'Loaded all {len(state)} state entries with strict=True.', flush=True)
    validation = checkpoint.get('validation')
    epoch = checkpoint.get('epoch')
    del checkpoint, state, mat_state
    model.to(args.device).eval()

    with Image.open(args.image) as source:
        original = source.convert('RGB')
    # Match load_pair/evaluate in the training notebook: PIL bilinear square
    # resize, RGB ImageNet normalization, resize logits before sigmoid.
    start = time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
        start = time.perf_counter()
    resized = original.resize((size, size), Image.Resampling.BILINEAR)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    normalized = (np.array(resized, dtype=np.float32) - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
    if args.config:
        from mmseg.smoke_inference import prepare_image
        tensor, metadata = prepare_image(cfg, args.image, args.device)
    print(f'Predicting {original.width}x{original.height} image on {args.device}...', flush=True)
    with torch.no_grad():
        if args.config:
            probability = model.inference(tensor, metadata, rescale=True)[0, 0].cpu().numpy()
        else:
            logits = model.encode_decode(tensor, [{}])
            probability = F.interpolate(
                logits, size=(original.height, original.width), mode='bilinear',
                align_corners=False).sigmoid()[0, 0].cpu().numpy()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not np.isfinite(probability).all():
        raise ValueError('Prediction contains non-finite probabilities.')
    foreground = probability > threshold
    mask = Image.fromarray(foreground.astype(np.uint8) * 255)
    pixels = np.array(original)
    overlay = pixels.copy()
    overlay[foreground] = np.rint(
        pixels[foreground] * 0.55 + np.array([255, 45, 45]) * 0.45).astype(np.uint8)
    overlay = Image.fromarray(overlay)
    stem = args.image.stem
    paths = {kind: args.output_dir / f'{stem}_{kind}.{extension}'
             for kind, extension in [('mask', 'png'), ('overlay', 'png'),
                                     ('probability', 'npy'), ('preview', 'jpg'),
                                     ('report', 'json')]}
    mask.save(paths['mask'])
    overlay.save(paths['overlay'])
    np.save(paths['probability'], probability)
    crop_debug = None
    if getattr(model, 'last_crop_debug', []):
        from mmseg.smoke_inference import restore_count_map
        debug = model.last_crop_debug[0]
        counts = restore_count_map(debug, metadata[0]).cpu().numpy()
        paths['count_map'] = args.output_dir / f'{stem}_count_map.npy'
        paths['uncovered'] = args.output_dir / f'{stem}_uncovered.png'
        np.save(paths['count_map'], counts)
        Image.fromarray((counts == 0).astype(np.uint8) * 255).save(paths['uncovered'])
        crop_debug = dict(scores=None if debug['scores'] is None else debug['scores'].cpu().tolist(),
                          selected=debug['selected'], fallback=debug['fallback'],
                          boxes=debug['boxes'], working_size=[512, 512],
                          resized_shape=metadata[0]['img_shape'],
                          uncovered_fraction=float((counts == 0).mean()))
    panel_width = min(original.width, 640)
    panel_height = round(original.height * panel_width / original.width)
    preview = Image.new('RGB', (panel_width * 3, panel_height + 32), '#20242b')
    draw = ImageDraw.Draw(preview)
    for i, (label, panel) in enumerate([
            ('Original', original), ('Smoke mask (white = smoke)', mask.convert('RGB')),
            (f'Overlay | threshold = {threshold}', overlay)]):
        draw.text((i * panel_width + 10, 9), label, fill='white')
        preview.paste(panel.resize((panel_width, panel_height),
                      Image.Resampling.NEAREST if i == 1 else Image.Resampling.BILINEAR),
                      (i * panel_width, 32))
    preview.save(paths['preview'], quality=95)
    report = dict(
        checkpoint=str(args.checkpoint.resolve()), image=str(args.image.resolve()),
        checkpoint_epoch_zero_based=epoch, strict_load=True, device=args.device,
        config=str(args.config) if args.config else None, crop_debug=crop_debug,
        input_size=[size, size], output_size=[original.width, original.height],
        threshold=threshold, mask_values={'background': 0, 'smoke': 255},
        smoke_pixels=int(foreground.sum()), total_pixels=int(foreground.size),
        smoke_fraction=float(foreground.mean()), inference_seconds=elapsed,
        checkpoint_validation=validation,
        image_evaluation=None,
        evaluation_note='No ground-truth mask supplied; IoU/F1/recall cannot be computed for this image. '
                        'checkpoint_validation refers to the saved validation dataset, not this image.',
        outputs={k: str(v.resolve()) for k, v in paths.items()})
    paths['report'].write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
