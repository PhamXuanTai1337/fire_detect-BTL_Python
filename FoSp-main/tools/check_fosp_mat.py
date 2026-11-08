"""Run a real FoSp forward/backward with pretrained MAT and synthetic labels.

Backbone pretrained download is disabled for this wiring test only. This is not
training for accuracy; the actual training config still uses pretrained MiT-B3.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mmcv import Config
from mmcv.utils import get_logger
from mmseg.models import build_segmentor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--output', type=Path, default=ROOT / 'work_dirs/mat_check/fosp_report.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    get_logger('mmcv', log_level=logging.WARNING)
    cfg = Config.fromfile(str(ROOT / 'config/fosp_config.py'))
    cfg.model.pretrained = None
    cfg.model.decode_head.mat_checkpoint = str(ROOT / 'pretrained/Places_512_FullData.pkl')
    cfg.model.decode_head.mat_impl = 'ref'
    print('Building FoSp + pretrained full MAT...', flush=True)
    model = build_segmentor(cfg.model)
    generator = model.decode_head.G.generator
    pretrained_weight = generator.synthesis.first_stage.conv_first.conv.weight.detach().clone()
    model.init_weights()
    assert torch.equal(pretrained_weight, generator.synthesis.first_stage.conv_first.conv.weight)
    model.to(args.device).train()
    assert not generator.training
    # Test the RGB bridge against the normalization defined in the config.
    rgb = torch.tensor([10., 80., 200.], device=args.device).view(1, 3, 1, 1)
    mean = rgb.new_tensor(cfg.img_norm_cfg['mean']).view(1, 3, 1, 1)
    std = rgb.new_tensor(cfg.img_norm_cfg['std']).view(1, 3, 1, 1)
    assert torch.allclose(model._renormalization((rgb-mean)/std,
                         cfg.img_norm_cfg['mean'], cfg.img_norm_cfg['std']), rgb/127.5-1)
    image = torch.rand(1, 3, args.size, args.size, device=args.device) * 255
    image = (image - mean) / std
    labels = torch.zeros(1, 1, args.size, args.size, dtype=torch.long, device=args.device)
    labels[:, :, args.size//4:3*args.size//4, args.size//4:3*args.size//4] = 1
    print('Running full FoSp forward/backward at {}x{}...'.format(args.size, args.size), flush=True)
    losses = model.forward_train(image, [dict()], labels)
    loss = sum(value.mean() for name, value in losses.items() if 'loss' in name)
    assert torch.isfinite(loss)
    loss.backward()
    checks = {
        'backbone': model.backbone,
        'bcg': model.decode_head.cls_seg_1_4,
        'projection': model.decode_head.foreground_channel_convs,
        'fusion': model.decode_head.fusion_foreground_inputs_conv,
        'classification': model.decode_head.fusion_conv_seg,
    }
    for name, module in checks.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), name
        assert any(g.abs().max() > 0 for g in gradients), name
    assert all(p.grad is None and not p.requires_grad for p in generator.parameters())
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        logits = model.encode_decode(image, [dict()])
    assert logits.shape == (1, 1, args.size, args.size)
    assert torch.isfinite(logits).all()
    report = dict(size=args.size, device=args.device, loss=float(loss.detach()),
                  gradient_checks=list(checks), mat_frozen=True,
                  pretrained_survives_init=True, rgb_bridge_correct=True,
                  logits_shape=list(logits.shape),
                  note='Synthetic labels and random backbone: wiring check, not accuracy.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
