"""Small tensor contract tests; these do not measure smoke detection quality."""
import importlib

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mmseg.datasets.pipelines.lab_clahe import LabCLAHE, apply_lab_clahe
from mmseg.models.decode_heads.base_decode_head import BaseDecodeHead
from mmseg.models.losses import BinaryTverskyLoss, WeightedBCELoss


@pytest.mark.parametrize('kind', ['tiny_smoke', 'background', 'ignore'])
def test_losses_gradients_and_ignore(kind):
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    label = torch.zeros(2, 1, 8, 8, dtype=torch.long)
    label[1] = 255
    if kind == 'tiny_smoke':
        label[0, 0, 2, 3] = 1
    elif kind == 'ignore':
        label[:] = 255
    loss = WeightedBCELoss()(logits, label) + BinaryTverskyLoss()(logits, label)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].eq(0).all()
    if kind == 'ignore':
        assert loss.item() == 0
        assert logits.grad.eq(0).all()
    else:
        assert logits.grad[0, 0, 0, 0] > 0
        if kind == 'tiny_smoke':
            assert logits.grad[0, 0, 2, 3] < 0


def test_loss_reductions_match_formulas():
    logits = torch.tensor([[[[0., 0.]]], [[[0., 0.]]], [[[0., 0.]]]])
    label = torch.tensor([[[[1, 255]]], [[[0, 0]]], [[[255, 255]]]])
    # Three valid pixels: one positive of weight four and two negatives.
    bce = WeightedBCELoss()(logits, label)
    assert bce.item() == pytest.approx(0.5 * 6 * np.log(2) / 3)
    eps = 1e-6
    positive_ti = (0.5 + eps) / (0.5 + 0.65 * 0.5 + eps)
    background_ti = eps / (0.35 + eps)
    expected = 0.5 * ((1 - positive_ti) + (1 - background_ti)) / 2
    assert BinaryTverskyLoss()(logits, label).item() == pytest.approx(expected)


class MultipleLossHead(BaseDecodeHead):
    def forward(self, inputs, origin_img):
        return inputs[0], inputs[1:]


def test_both_losses_reach_all_six_heads():
    head = MultipleLossHead(
        in_channels=1, channels=1, num_classes=2, out_channels=1, threshold=0.4,
        loss_decode=[dict(type='WeightedBCELoss'), dict(type='BinaryTverskyLoss')])
    logits = [torch.randn(1, 1, 4, 4, requires_grad=True) for _ in range(6)]
    labels = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    labels[0, 0, 0, 0] = 1
    result = head.forward_train(logits, None, [], labels, {})
    assert set(result) == {'loss_weighted_bce', 'loss_tversky', 'acc_seg'}
    for name in ('loss_weighted_bce', 'loss_tversky'):
        expected = 0.5 * head.losses(logits[0], labels)[name] + sum(
            0.1 * head.losses(x, labels)[name] for x in logits[1:])
        torch.testing.assert_close(result[name], expected)
        grads = torch.autograd.grad(result[name], logits, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_clahe_color_order_mask_and_single_application():
    rgb = np.random.default_rng(4).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3., tileGridSize=(16, 16)).apply(lab[:, :, 0])
    expected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    np.testing.assert_array_equal(apply_lab_clahe(rgb, 'RGB'), expected)
    np.testing.assert_array_equal(apply_lab_clahe(rgb[:, :, ::-1], 'BGR')[:, :, ::-1], expected)
    mask = np.zeros((64, 96), dtype=np.uint8)
    transform = LabCLAHE(color_order='RGB')
    results = transform(dict(img=rgb, gt_semantic_seg=mask))
    assert results['gt_semantic_seg'] is mask
    with pytest.raises(ValueError, match='already applied'):
        transform(results)
    with pytest.raises(ValueError, match='before Normalize'):
        transform(dict(img=rgb, img_norm_cfg={}))
    with pytest.raises(ValueError, match='uint8'):
        apply_lab_clahe(rgb.astype(np.float32) / 255, 'RGB')


class FeatureStubMAT(nn.Module):
    """Feature-shape stand-in only; real MAT freeze tests live in test_full_mat."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls = 0

    def forward(self, image, mask):
        self.calls += 1
        features = [image.new_zeros(image.shape[0], c, s, s)
                    for c, s in zip((512, 512, 512, 256), (1, 2, 4, 8))]
        return features, features


def test_focus_skips_mat_and_extra_branch_backpropagates(monkeypatch, tmp_path):
    module = importlib.import_module('mmseg.models.decode_heads.FOSP_main')
    monkeypatch.setattr(module, 'FullMATInpainter', FeatureStubMAT)
    options = dict(in_channels=[2, 4, 8, 16], in_index=[0, 1, 2, 3], channels=4,
                   num_classes=2, out_channels=1, threshold=0.4, norm_cfg=None,
                   shallow_fusion=True, dropout_ratio=0)
    head = module.FOSP(**options)
    backbone = nn.Conv2d(3, 2, 3, stride=4, padding=1)
    shallow = backbone(torch.randn(1, 3, 32, 32))
    features = [shallow] + [F.avg_pool2d(shallow, scale).repeat(1, scale, 1, 1)
                            for scale in (2, 4, 8)]
    assert features[0].shape[-2:] == (8, 8)
    head.forward_focus(features)
    assert head.G.calls == 0
    logits, _ = head(features, torch.randn(1, 3, 512, 512))
    assert head.G.calls == 1
    logits.square().mean().backward()
    for part in (head.shallow_projection, head.shallow_fusion_conv, backbone):
        grads = [p.grad for p in part.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    path = tmp_path / 'head.pth'
    torch.save(head.state_dict(), path)
    restored = module.FOSP(**options)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
