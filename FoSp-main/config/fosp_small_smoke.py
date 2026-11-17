"""Full MAT + small-smoke experiments; no training initialization is implicit.

Set FOSP_CROP_MODE=whole|all|selective. Independent ablations use environment
flags FOSP_CLAHE, FOSP_SMALL_SMOKE_LOSS, FOSP_SHALLOW_FUSION, FOSP_SELECTOR,
FOSP_TRAIN_CROPS (0/1) before loading this config. fosp_kaggle.py stays baseline.
Train with --initialization pretrained, or finetune --load-from PATH,
or resume --resume-from PATH. No old optimizer is used for finetuning.
"""
import os

_base_ = ['./fosp_kaggle.py']

clahe_enabled = os.environ.get('FOSP_CLAHE', '1') == '1'
small_smoke_loss_enabled = os.environ.get('FOSP_SMALL_SMOKE_LOSS', '1') == '1'
shallow_fusion_enabled = os.environ.get('FOSP_SHALLOW_FUSION', '1') == '1'
selector_enabled = os.environ.get('FOSP_SELECTOR', '1') == '1'
train_crops_enabled = os.environ.get('FOSP_TRAIN_CROPS', '1') == '1'
crop_mode = os.environ.get('FOSP_CROP_MODE', 'selective')
require_explicit_initialization = True
initialization = None
load_from = None
resume_from = None

model = dict(
    type='SelectiveCropFOSP',
    crop_cfg=dict(mode=crop_mode, threshold=0.5, relative_gap=0.10,
                  selector_loss_weight=1.0, train_crops=train_crops_enabled),
    decode_head=dict(shallow_fusion=shallow_fusion_enabled,
                     crop_selector=selector_enabled, selector_channels=32, threshold=0.4))
if small_smoke_loss_enabled:
    model['decode_head']['loss_decode'] = [
        dict(type='WeightedBCELoss', pos_weight=4.0, loss_weight=0.5),
        dict(type='BinaryTverskyLoss', alpha=0.35, beta=0.65, loss_weight=0.5)]

_clahe = [dict(type='LabCLAHE', clip_limit=3.0, tile_grid_size=(16, 16),
               color_order='BGR')] if clahe_enabled else []
_normalize = dict(type='Normalize', mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375], to_rgb=True)
# Keep the existing augmentation order. CLAHE runs after photometric distortion
# on this augmented panorama, before letterboxing, Normalize and the nine crops.
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5), dict(type='PhotoMetricDistortion'),
] + _clahe + [
    dict(type='SmokeLetterbox'), _normalize,
    dict(type='DefaultFormatBundle'), dict(type='Collect', keys=['img', 'gt_semantic_seg'])]
test_pipeline = [dict(type='LoadImageFromFile')] + _clahe + [
    dict(type='MultiScaleFlipAug', img_scale=(512, 512), flip=False, transforms=[
        dict(type='SmokeLetterbox'), dict(type='RandomFlip'), _normalize,
        dict(type='ImageToTensor', keys=['img']), dict(type='Collect', keys=['img'])])]
data = dict(train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
find_unused_parameters = True  # Legacy unused heads and all-ignore batches.
work_dir = '/kaggle/working/fosp_small_smoke'
