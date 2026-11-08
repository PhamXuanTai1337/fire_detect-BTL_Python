"""Contract tests for checkpoint loading and paired frozen MAT execution."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mat_inpainter.adapter import FullMATInpainter, load_pretrained


class TinyStochasticGenerator(nn.Module):
    """Exercise the adapter's RNG contract without constructing a large MAT."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, image, mask, z, label, **kwargs):
        self.calls.append((z.clone(), mask.clone(), self.training))
        feature = F.avg_pool2d(image * mask, 32)
        feature = F.dropout(feature, training=True) * self.weight
        return image, [feature] * 4


@pytest.fixture
def tiny_adapter():
    adapter = FullMATInpainter.__new__(FullMATInpainter)
    nn.Module.__init__(adapter)
    adapter.seed = 240
    adapter.noise_mode = 'const'
    adapter.generator = TinyStochasticGenerator()
    adapter.requires_grad_(False)
    adapter.eval()
    return adapter


def test_checkpoint_is_mandatory(tmp_path):
    with pytest.raises(ValueError, match='required'):
        FullMATInpainter(None)
    with pytest.raises(FileNotFoundError, match='Missing MAT'):
        FullMATInpainter(tmp_path / 'missing.pkl')


def test_checkpoint_loading_is_strict(tmp_path):
    layer = nn.Linear(3, 2)
    path = tmp_path / 'weights.pth'
    expected = {k: torch.full_like(v, 0.25) for k, v in layer.state_dict().items()}
    torch.save(expected, path)
    load_pretrained(layer, path)
    assert all(torch.equal(layer.state_dict()[k], v) for k, v in expected.items())
    torch.save({'weight': expected['weight']}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_pretrained(layer, path)


def test_parent_train_keeps_mat_frozen(tiny_adapter):
    parent = nn.Sequential(tiny_adapter)
    parent.train()
    assert not tiny_adapter.training
    assert not tiny_adapter.generator.training
    assert all(not p.requires_grad for p in tiny_adapter.parameters())


def test_equal_masks_cancel_and_rng_is_preserved(tiny_adapter):
    image = torch.randn(1, 3, 512, 512)
    mask = torch.ones(1, 1, 512, 512)
    rng = torch.get_rng_state().clone()
    a, b = tiny_adapter(image, mask)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(not x.requires_grad for x in a + b)
    calls = tiny_adapter.generator.calls
    assert len(calls) == 2
    assert torch.equal(calls[0][0], calls[1][0])


def test_changed_mask_reaches_only_masked_branch(tiny_adapter):
    image = torch.ones(1, 3, 512, 512)
    mask = torch.ones(1, 1, 512, 512)
    mask[:, :, 128:384, 128:384] = 0
    a, b = tiny_adapter(image, mask)
    assert not torch.equal(a[0], b[0])
    assert torch.equal(tiny_adapter.generator.calls[0][1], mask)
    assert tiny_adapter.generator.calls[1][1].eq(1).all()
    again, _ = tiny_adapter(image, mask)
    assert torch.equal(a[0], again[0])


def test_bad_shapes_fail_explicitly(tiny_adapter):
    with pytest.raises(ValueError, match='512'):
        tiny_adapter(torch.zeros(1, 3, 256, 256), torch.ones(1, 1, 256, 256))
