import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config

from mmseg.datasets.pipelines import Compose
from mmseg.datasets.pipelines.smoke_letterbox import SmokeLetterbox
from mmseg.models.segmentors.encoder_decoder_fosp import EncoderDecoderFOSP
from mmseg.models.segmentors.selective_crop_fosp import SelectiveCropFOSP
from mmseg.smoke_checkpoint import load_smoke_weights, validate_initialization
from mmseg.smoke_crops import (CROP_BOXES, CropSelectorHead, crop_targets, extract_crop,
                              merge_crop_probabilities, sample_training_crops, select_crops)
from mmseg.smoke_metrics import independent_splits, measure_image, summarize


def test_geometry_and_mask_pair():
    assert len(CROP_BOXES) == len(set(CROP_BOXES)) == 9
    assert CROP_BOXES[0] == (0, 0, 256, 256)
    assert CROP_BOXES[4] == (128, 128, 384, 384)
    assert CROP_BOXES[8] == (256, 256, 512, 512)
    mask = torch.zeros(1, 1, 512, 512, dtype=torch.long)
    mask[..., 128, 128] = 1
    crop = extract_crop(mask, 4, is_mask=True)
    assert crop[..., :2, :2].eq(1).all() and crop.sum() == 4
    image = mask.float().expand(-1, 3, -1, -1)
    assert extract_crop(image, 4)[..., 0, 0].eq(1).all()
    with pytest.raises(ValueError, match='512'):
        extract_crop(mask[..., :128, :], 0)


@pytest.mark.parametrize('scores,threshold,expected,fallback', [
    ([.7] * 9, .5, list(range(9)), False),
    ([.5, .49, .51, 0, 0, 0, 0, 0, 0], .5, [0, 2], False),
    ([.4, .36, .1, 0, 0, 0, 0, 0, 0], .5, [0, 1], True),
    ([.4, .35, .1, 0, 0, 0, 0, 0, 0], .5, [0], True),
    ([.1, .2, .2, 0, 0, 0, 0, 0, 0], .5, [1, 2], True),
    ([0.] * 9, .5, [0, 1], True),
])
def test_selection(scores, threshold, expected, fallback):
    assert select_crops(torch.tensor(scores), threshold) == (expected, fallback)


def test_probability_merge_overlap_and_uncovered():
    indices = [0, 1, 3, 4]
    probs = [torch.full((1, 1, 512, 512), value) for value in (.1, .3, .5, .9)]
    result, counts = merge_crop_probabilities(probs, indices)
    assert counts[0, 0, 10, 10] == 1
    assert result[0, 0, 10, 10] == pytest.approx(.1)
    assert counts[0, 0, 10, 150] == 2
    assert result[0, 0, 10, 150] == pytest.approx(.2)
    assert counts[0, 0, 150, 150] == 4
    assert result[0, 0, 150, 150] == pytest.approx(.45)
    assert counts[0, 0, 500, 500] == result[0, 0, 500, 500] == 0
    assert torch.isfinite(result).all()
    # Omitted crops never dilute another crop's prediction.
    only, count = merge_crop_probabilities([probs[3]], [4])
    assert only[0, 0, 150, 150] == pytest.approx(.9)
    assert count.max() == 1


def test_targets_ignore_and_sampling_do_not_depend_on_selector():
    mask = torch.zeros(3, 1, 512, 512, dtype=torch.long)
    mask[0, 0, 1, 1] = 1
    mask[2] = 255
    labels, valid = crop_targets(mask)
    assert labels[0].tolist() == [1] + [0] * 8
    assert not valid[2].any()
    torch.manual_seed(42)
    selected = sample_training_crops(labels, valid)
    assert selected[0][0] == 0 and selected[0][1] != 0
    assert len(set(selected[1])) == 2 and selected[2] == []
    torch.manual_seed(42)
    assert selected == sample_training_crops(labels, valid)


def test_selector_gradients_and_independent_logits():
    head = CropSelectorHead(4)
    shallow = torch.randn(2, 4, 128, 128, requires_grad=True)
    focus = torch.randn(2, 1, 32, 32, requires_grad=True)
    logits = head(shallow, focus)
    assert logits.shape == (2, 9)
    F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert shallow.grad.abs().sum() > 0 and focus.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_letterbox_preserves_aspect_labels_and_metadata():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[10, 20] = 1
    result = SmokeLetterbox()(dict(img=image, gt_semantic_seg=mask,
                                   seg_fields=['gt_semantic_seg'], ori_shape=image.shape))
    assert result['img'].shape == (512, 512, 3)
    assert result['img_shape'] == (256, 512, 3)
    assert result['ori_shape'] == (200, 400, 3)
    assert set(np.unique(result['gt_semantic_seg'])) == {0, 1, 255}
    assert np.all(result['gt_semantic_seg'][256:] == 255)
    np.testing.assert_allclose(result['scale_factor'], [1.28] * 4)


class Harness(SelectiveCropFOSP):
    """Exercise routing, batch, merge and restoration without full MAT cost."""
    def __init__(self):
        nn.Module.__init__(self)
        self.crop_cfg = dict(mode='selective', threshold=.5, relative_gap=.1,
                             train_crops=True, selector_loss_weight=1.)
        self.focus_calls = self.full_calls = 0
        self.focus_parameter = nn.Parameter(torch.tensor([2., -4., -4., -4., -4., -4., -4., -4., -4.]))
        self.seg_parameter = nn.Parameter(torch.tensor(0.))
        self.decode_head = SimpleNamespace(selector_head=True, ignore_index=255, threshold=.4)
        self.decode_head.loss_decode = SimpleNamespace(loss_name='loss_seg')
        self.out_channels = 1

    def forward_focus(self, img):
        self.focus_calls += 1
        return self.focus_parameter.unsqueeze(0).expand(img.shape[0], -1)

    def encode_decode_full(self, img, img_metas):
        self.full_calls += 1
        return self.seg_parameter.expand(img.shape[0], 1, 512, 512)


def test_route_batch_restore_and_exact_zero():
    model = Harness()
    image = torch.zeros(2, 3, 512, 512)
    meta = [dict(img_shape=(256, 512, 3), ori_shape=(200, 400, 3), flip=False)] * 2
    probability = model.inference(image, meta, True)
    assert model.focus_calls == 1 and model.full_calls == 2
    assert probability.shape == (2, 1, 200, 400)
    assert probability[..., :190].eq(.5).all()
    assert probability[..., 210:].eq(0).all()
    assert model.last_crop_debug[0]['selected'] == [0]
    assert model.last_crop_debug[0]['count_map'].sum() == 256 * 256
    model.predict_probability(image[:1], meta[:1], mode='all')
    assert model.focus_calls == 1 and model.full_calls == 11
    assert model.last_crop_debug[0]['count_map'].min() >= 1
    with torch.no_grad():
        pred, probs = model.simple_test(image[:1], meta[:1], True)
    assert len(pred) == len(probs) == 1
    assert pred[0].shape == (200, 400)


def test_training_covers_missed_positive_and_keeps_losses_separate(monkeypatch):
    model = Harness()
    captured = []
    def fake_train(self, image, metadata, labels):
        captured.append(labels)
        return {'decode.loss_seg': (self.seg_parameter - labels.float().mean()).square()}
    monkeypatch.setattr(EncoderDecoderFOSP, 'forward_train', fake_train)
    image = torch.zeros(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512, dtype=torch.long)
    mask[..., 500, 500] = 1  # Selector favors crop 0, but positive is crop 8.
    losses = model.forward_train(image, [{}], mask)
    assert len(captured) == 2 and captured[0].sum() == 4 and captured[1].sum() == 0
    assert 'selector.loss_bce' in losses and 'decode.loss_seg' in losses
    sum(v for k, v in losses.items() if 'loss' in k).backward()
    assert model.focus_parameter.grad.abs().sum() > 0
    assert model.seg_parameter.grad.abs().sum() > 0
    captured.clear()
    result = model.forward_train(image, [{}], torch.full_like(mask, 255))
    assert not captured and result['selector.loss_bce'] == 0


def test_checkpoint_migration_only_permits_new_modules():
    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.decode_head = nn.Module()
    model.decode_head.selector_head = nn.Linear(2, 1)
    state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('decode_head.')}
    report = load_smoke_weights(model, state, allow_new_modules=True)
    assert report['missing'] == ['decode_head.selector_head.bias', 'decode_head.selector_head.weight']
    with pytest.raises(ValueError, match='missing'):
        load_smoke_weights(model, state)
    with pytest.raises(ValueError, match='backbone.weight'):
        load_smoke_weights(model, {'backbone.bias': state['backbone.bias']}, True)
    with pytest.raises(ValueError, match='unexpected'):
        load_smoke_weights(model, {**state, 'oops': torch.zeros(1)}, True)
    with pytest.raises(ValueError, match='shape_mismatch'):
        load_smoke_weights(model, {**state, 'backbone.weight': torch.zeros(3, 2)}, True)


def test_config_and_initialization_guard():
    cfg = Config.fromfile('config/fosp_small_smoke.py')
    with pytest.raises(ValueError, match='Choose'):
        validate_initialization(cfg)
    cfg.initialization = 'finetune'
    with pytest.raises(ValueError, match='requires'):
        validate_initialization(cfg)
    cfg.load_from = 'explicit_checkpoint.pth'
    validate_initialization(cfg)
    assert cfg.runner.max_iters == 40000
    assert cfg.model.decode_head.threshold == .4
    assert [x.type for x in cfg.data.train.pipeline].index('PhotoMetricDistortion') < [x.type for x in cfg.data.train.pipeline].index('LabCLAHE')
    for name in ('train', 'val', 'test'):
        Compose(cfg.data[name].pipeline)


def test_actual_image_pipeline_applies_clahe_once_and_matches_rgb(tmp_path, monkeypatch):
    import cv2
    from PIL import Image
    import mmseg.datasets.pipelines.lab_clahe as clahe_module
    from mmseg.smoke_inference import prepare_image
    cfg = Config.fromfile('config/fosp_small_smoke.py')
    rgb = np.random.default_rng(7).integers(0, 256, (200, 400, 3), dtype=np.uint8)
    path = tmp_path / 'test.png'
    Image.fromarray(rgb).save(path)
    calls = []
    original = clahe_module.apply_lab_clahe
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(clahe_module, 'apply_lab_clahe', counted)
    tensor, meta = prepare_image(cfg, path, 'cpu')
    assert len(calls) == 1
    expected = original(rgb, 'RGB')
    expected = cv2.resize(expected, (512, 256))
    restored = tensor[0].permute(1, 2, 0).numpy() * np.array([58.395, 57.12, 57.375]) + np.array([123.675, 116.28, 103.53])
    np.testing.assert_allclose(restored[:256], expected, atol=3e-5)
    np.testing.assert_allclose(restored[256:], 0, atol=3e-5)
    assert meta[0]['img_shape'][:2] == (256, 512)
    # Exercise the actual train pipeline, including loader identity metadata.
    Image.fromarray(np.zeros((200, 400), dtype=np.uint8)).save(tmp_path / 'mask.png')
    train = Compose(cfg.data.train.pipeline)(dict(
        img_info=dict(filename=str(path)), ann_info=dict(seg_map=str(tmp_path / 'mask.png')),
        img_prefix=None, seg_prefix=None, seg_fields=[]))
    assert train['img'].data.shape == (3, 512, 512)
    assert train['gt_semantic_seg'].data.shape == (1, 512, 512)
    assert len(calls) == 2


def test_metrics_coverage_negative_and_ignore(tmp_path):
    mask = np.array([[1, 0], [255, 0]], dtype=np.uint8)
    probability = np.array([[0., .9], [1., 0.]])
    covered = np.array([[False, True], [True, False]])
    positive = measure_image(probability, mask, covered)
    assert (positive['tp'], positive['fp'], positive['fn']) == (0, 1, 1)
    assert positive['covered_smoke_px'] == positive['covered_regions'] == 0
    negative = measure_image(probability, np.zeros((2, 2), dtype=np.uint8), covered)
    rows = [dict(r, crop_count=1, fallback=False, latency_seconds=.1) for r in (positive, negative)]
    summary = summarize(rows)
    assert summary['negative_false_alarm_rate'] == 1
    assert summary['positive_no_tp_rate'] == 1
    assert summary['smoke_pixel_coverage'] == 0
    json.dumps(summary, allow_nan=False)
    data = {}
    for name in ('train', 'val', 'test'):
        (tmp_path / f'{name}.txt').write_text(name)
        data[name] = dict(data_root=str(tmp_path), split=name + '.txt')
    independent_splits(data)
    (tmp_path / 'test.txt').write_text('train')
    with pytest.raises(ValueError, match='overlap'):
        independent_splits(data)
