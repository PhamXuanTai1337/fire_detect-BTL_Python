"""Overfit FoSp + full MAT on 32 Pyro-SDIS box masks for 50 epochs.

Run: python run_fit_test.py
Prepare only: python run_fit_test.py --prepare-only
Resume: python run_fit_test.py --resume

This is a memorization/wiring test, NOT a segmentation benchmark. Train and
evaluation deliberately use the same images. Filled rectangles are proxy masks;
IoU measures agreement with those rectangles, not the true smoke boundary.
Pyro-SDIS v2 class 0 = smoke:
https://huggingface.co/datasets/pyronear/pyro-sdis/blob/main/README.md
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_DATA = Path(
    r'C:\Users\User\OneDrive\Documents\AI_ICN\Data set\fire_detect\pyro-sdis\data\unpack')


def read_boxes(path, smoke_class=0):
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{line_number}: expected YOLO class xc yc w h')
        class_id, xc, yc, width, height = map(float, parts)
        if not all(math.isfinite(v) for v in (class_id, xc, yc, width, height)):
            raise ValueError(f'{path}:{line_number}: non-finite label')
        if class_id != int(class_id) or not (0 <= xc <= 1 and 0 <= yc <= 1
                                            and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f'{path}:{line_number}: invalid normalized box')
        if int(class_id) == smoke_class:
            boxes.append((xc, yc, width, height))
    return boxes


def rasterize_boxes(boxes, size=512):
    import numpy as np
    mask = np.zeros((size, size), dtype=np.uint8)
    for xc, yc, width, height in boxes:
        left = max(0, math.floor((xc - width / 2) * size))
        top = max(0, math.floor((yc - height / 2) * size))
        right = min(size, math.ceil((xc + width / 2) * size))
        bottom = min(size, math.ceil((yc + height / 2) * size))
        mask[top:bottom, left:right] = 1
    return mask


def prepare_data(args, output):
    from PIL import Image
    source = args.data_root.resolve()
    labels = sorted((source / 'labels/train').glob('*.txt'))
    if not labels:
        raise ValueError(f'No YOLO labels under {source / "labels/train"}')
    random.Random(args.seed).shuffle(labels)
    selected = []
    for label in labels:
        boxes = read_boxes(label, args.smoke_class)
        if not boxes:
            continue
        image = next((source / 'images/train' / (label.stem + ext)
                      for ext in ('.jpg', '.jpeg', '.png', '.webp')
                      if (source / 'images/train' / (label.stem + ext)).is_file()), None)
        if image is None:
            raise FileNotFoundError(f'Image missing for {label}')
        selected.append(dict(id=label.stem, image=image.as_posix(),
                             label=label.as_posix(), boxes=boxes))
        if len(selected) == args.images:
            break
    if len(selected) != args.images:
        raise ValueError(f'Need {args.images} positive images, found {len(selected)}')
    manifest = dict(source=source.as_posix(), seed=args.seed, smoke_class=args.smoke_class,
                    images=args.images, size=512, label_type='filled_box_proxy',
                    evaluation='same_images_as_training', selected=selected)
    manifest_path = output / 'selected_images.json'
    if args.resume:
        if not manifest_path.exists():
            raise ValueError('Cannot resume without selected_images.json')
        if json.loads(manifest_path.read_text(encoding='utf-8')) != json.loads(json.dumps(manifest)):
            raise ValueError('Selection/data differs from previous run; use a new --work-dir.')
    data = output / 'data'
    for name in ('JPEGImages', 'SegmentationClass', 'ImageSets/Segmentation'):
        (data / name).mkdir(parents=True, exist_ok=True)
    for item in selected:
        with Image.open(item['image']) as image:
            image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR).save(
                data / 'JPEGImages' / (item['id'] + '.jpg'), quality=95)
        mask = rasterize_boxes(item['boxes'])
        if not mask.any():
            raise ValueError(f'Empty foreground after rasterization: {item["id"]}')
        Image.fromarray(mask).save(data / 'SegmentationClass' / (item['id'] + '.png'))
    split = '\n'.join(item['id'] for item in selected) + '\n'
    for name in ('train', 'val', 'test'):
        (data / 'ImageSets/Segmentation' / (name + '.txt')).write_text(split, encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return data


def create_config(args, output, data):
    from mmcv import Config
    cfg = Config.fromfile(str(ROOT / 'config/fosp_config.py'))
    cfg.model.decode_head.mat_checkpoint = args.mat_checkpoint.resolve().as_posix()
    cfg.model.decode_head.mat_impl = 'ref'
    cfg.model.backbone.drop_path_rate = 0.0
    cfg.model.decode_head.dropout_ratio = 0.0
    if args.backbone_checkpoint:
        cfg.model.pretrained = args.backbone_checkpoint.replace('\\', '/')
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0  # portable on Windows; dataset has only 32 images
    normalization = dict(type='Normalize', **dict(cfg.img_norm_cfg))
    train_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
                      dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
                      dict(type='RandomFlip', prob=0.0),
                      normalization, dict(type='DefaultFormatBundle'),
                      dict(type='Collect', keys=['img', 'gt_semantic_seg'])]
    eval_pipeline = [dict(type='LoadImageFromFile'),
                     dict(type='MultiScaleFlipAug', img_scale=(512, 512), flip=False,
                          transforms=[dict(type='Resize', keep_ratio=False),
                                      dict(type='RandomFlip'), normalization,
                                      dict(type='ImageToTensor', keys=['img']),
                                      dict(type='Collect', keys=['img'])])]
    for name in ('train', 'val', 'test'):
        cfg.data[name] = dict(type='SmokeDataset', data_root=data.as_posix(),
                             img_dir='JPEGImages', ann_dir='SegmentationClass',
                             split=f'ImageSets/Segmentation/{name}.txt',
                             pipeline=train_pipeline if name == 'train' else eval_pipeline)
    cfg.train_pipeline, cfg.test_pipeline = train_pipeline, eval_pipeline
    cfg.data_root = data.as_posix()
    cfg.runner.max_iters = args.images * args.epochs  # batch 1, no RepeatDataset
    cfg.lr_config = dict(policy='fixed', by_epoch=False)  # no 1500-step warmup
    cfg.optimizer.lr = args.lr
    cfg.optimizer.weight_decay = 0.0
    cfg.log_config.interval = args.images
    cfg.checkpoint_config = dict(by_epoch=False, interval=args.images * 5,
                                 max_keep_ckpts=2, create_symlink=False)
    cfg.evaluation = dict(interval=args.images, metric=['mIoU', 'mFscore'],
                          pre_eval=True, save_best='IoU.smoke', rule='greater')
    cfg.work_dir = output.as_posix()
    cfg.cudnn_benchmark = False
    cfg.load_from = cfg.resume_from = None
    cfg.gpu_ids = [0]
    config_path = output / 'fit_config.py'
    config_path.write_text(cfg.pretty_text, encoding='utf-8')
    return config_path


def export_history(output, images):
    rows = {}
    for path in sorted(output.glob('*.log.json')):
        last_train = None
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            if item.get('mode') == 'train':
                last_train = item
            elif item.get('mode') == 'val' and last_train:
                step = last_train['iter']
                rows[step] = dict(epoch=step / images, iteration=step,
                                  loss=last_train.get('loss'),
                                  box_mask_IoU=item['IoU.smoke'],
                                  box_mask_F1=item['Fscore.smoke'])
    if rows:
        values = [rows[step] for step in sorted(rows)]
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        print(f'History: {output / "history.csv"}')
        print(f'Best BOX-MASK IoU: {max(row["box_mask_IoU"] for row in values):.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--smoke-class', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2077)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work_dirs/fit_test_32')
    parser.add_argument('--mat-checkpoint', type=Path,
                        default=ROOT / 'pretrained/Places_512_FullData.pkl')
    parser.add_argument('--backbone-checkpoint', help='Optional local MiT-B3 checkpoint; otherwise auto-download.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.images <= 0 or args.epochs <= 0 or args.lr <= 0:
        parser.error('images, epochs and lr must be positive')
    if not args.mat_checkpoint.is_file():
        parser.error('MAT weight missing. Run tools/download_mat.py first.')
    output = args.work_dir.resolve()
    if output.exists() and list(output.glob('*.pth')) and not args.resume:
        parser.error('Checkpoints already exist. Use --resume or choose a fresh --work-dir.')
    output.mkdir(parents=True, exist_ok=True)
    print('OVERFIT TEST: same train/eval images; filled BOX masks, not true smoke masks.', flush=True)
    data = prepare_data(args, output)
    config = create_config(args, output, data)
    print(f'{args.images} images x {args.epochs} epochs = {args.images * args.epochs} optimizer steps.', flush=True)
    print(f'Config: {config}', flush=True)
    print('Full MAT frozen; augmentation/dropout/warmup disabled; batch size 1.', flush=True)
    if args.prepare_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required for this full MAT fit test.')
    environment = os.environ.copy()
    environment.setdefault('TORCH_HOME', str(ROOT / 'pretrained/torch_cache'))
    command = [sys.executable, str(ROOT / 'tools/train.py'), str(config), '--seed', str(args.seed)]
    if args.resume:
        command.append('--auto-resume')
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    finally:
        export_history(output, args.images)


if __name__ == '__main__':
    main()
