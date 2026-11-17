"""One synthetic full-MAT forward/backward and strict checkpoint round-trip.

Execution/gradient check only: no optimizer step and no quality measurement.
Uses a random backbone to avoid selecting a training initialization policy.
"""
import argparse
from pathlib import Path
import sys
import tempfile

import torch
from mmcv import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmseg.models import build_segmentor
from mmseg.core import build_optimizer
from mmseg.smoke_checkpoint import load_smoke_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/fosp_small_smoke.py')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mat-checkpoint', type=Path, required=True)
    parser.add_argument('--compatibility-checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.decode_head.mat_checkpoint = str(args.mat_checkpoint.resolve())
    cfg.model.decode_head.mat_impl = 'ref'
    model = build_segmentor(cfg.model).to(args.device)
    model.train()
    assert not model.decode_head.G.training
    assert all(not p.requires_grad for p in model.decode_head.G.parameters())
    optimizer = build_optimizer(model, cfg.optimizer)
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in parameter_ids for p in model.parameters() if p.requires_grad)
    del optimizer
    if args.compatibility_checkpoint:
        state = torch.load(args.compatibility_checkpoint, map_location='cpu', weights_only=True)['state_dict']
        # Audit compatibility without adopting these weights for the dry run.
        current = model.state_dict()
        allowed = ('decode_head.selector_head.', 'decode_head.shallow_projection.',
                   'decode_head.shallow_fusion_conv.')
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        mismatch = [k for k in current.keys() & state.keys() if current[k].shape != state[k].shape]
        assert not unexpected and not mismatch and all(k.startswith(allowed) for k in missing)
        print(f'Compatibility audit: missing new modules={missing}; unexpected={unexpected}; shape mismatch={mismatch}', flush=True)
        del current, state
    image = torch.randn(1, 3, 512, 512, device=args.device)
    mask = torch.zeros(1, 1, 512, 512, dtype=torch.long, device=args.device)
    mask[..., 16, 16] = 1
    mask[..., 500:, :] = 255
    calls = []
    handle = model.decode_head.G.register_forward_hook(lambda *unused: calls.append(1))
    with torch.no_grad():
        scores = model.forward_focus(image)
    assert scores.shape == (1, 9) and not calls
    print('Focus-only: PASS (zero MAT calls). Running two full training crops...', flush=True)
    losses = model.forward_train(image, [{}], mask)
    assert len(calls) == 2
    total = sum(value for key, value in losses.items() if 'loss' in key)
    assert torch.isfinite(total)
    total.backward()
    for name, module in [('selector', model.decode_head.selector_head),
                         ('shallow_projection', model.decode_head.shallow_projection),
                         ('shallow_fusion', model.decode_head.shallow_fusion_conv),
                         ('backbone', model.backbone)]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
        print(f'{name} gradients: PASS', flush=True)
    assert all(p.grad is None for p in model.decode_head.G.parameters())
    handle.remove()
    del total, losses, scores
    model.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(prefix='fosp_check_') as temporary:
        path = Path(temporary) / 'state.pth'
        torch.save(model.state_dict(), path)
        state = torch.load(path, map_location=args.device, weights_only=True)
        report = load_smoke_weights(model, state)
        assert not report['missing'] and not report['unexpected']
    print('Full MAT forward/backward + strict save/load: PASS. No optimizer step or quality evaluation.', flush=True)


if __name__ == '__main__':
    main()
