"""Real pretrained MAT smoke test, including both complete stages and RNG pairing."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from mat_inpainter import FullMATInpainter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'pretrained/Places_512_FullData.pkl'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--impl', choices=['ref', 'cuda'], default='ref')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'work_dirs/mat_check')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print('Loading complete pretrained MAT...', flush=True)
    model = FullMATInpainter(args.checkpoint, impl=args.impl).to(args.device)
    model.train()  # Must remain frozen/eval even when FoSp enters train mode.
    assert not model.generator.training
    assert all(not p.requires_grad for p in model.parameters())
    counts = dict(stage1=0, decoder512=0)

    def count(name):
        def hook(module, inputs, output):
            counts[name] += 1
        return hook

    handles = [
        model.generator.synthesis.first_stage.register_forward_hook(count('stage1')),
        model.generator.synthesis.dec.Dec_512x512.register_forward_hook(count('decoder512')),
    ]
    image_path = args.image or next((ROOT / 'data/SmokeSeg/JPEGImages').glob('*.jpg'))
    rgb = Image.open(image_path).convert('RGB').resize((512, 512))
    image = torch.from_numpy(np.array(rgb).copy()).permute(2, 0, 1).float()[None]
    image = (image / 127.5 - 1).to(args.device)
    mask = torch.ones(1, 1, 512, 512, device=args.device)
    mask[:, :, 192:320, 192:320] = 0
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(image.device).clone() if image.is_cuda else None
    start = time.perf_counter()
    print('Running masked + unmasked full MAT branches at 512x512...', flush=True)
    a, b, filled, original = model(image, mask, return_images=True)
    if image.is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    shapes = [(1, 512, 16, 16), (1, 512, 32, 32),
              (1, 512, 64, 64), (1, 256, 128, 128)]
    assert [tuple(x.shape) for x in a] == shapes
    assert [tuple(x.shape) for x in b] == shapes
    assert all(torch.isfinite(x).all() for x in a + b + [filled, original])
    assert counts == dict(stage1=2, decoder512=2), counts
    assert torch.equal(original, image), 'All-ones branch must preserve source RGB.'
    assert torch.equal(filled * mask, image * mask), 'Known pixels must be preserved.'
    assert any(not torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if image.is_cuda:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(image.device))
    print('Checking identical-mask cancellation...', flush=True)
    equal_a, equal_b = model(image, torch.ones_like(mask))
    # CUDA convolution can have tiny FP32 rounding differences even with paired
    # RNG. Do not confuse those with independent dropout/noise (large errors).
    same_mask_max_errors = [(x - y).abs().max().item() for x, y in zip(equal_a, equal_b)]
    assert all(torch.allclose(x, y, rtol=1e-5, atol=1e-4) for x, y in zip(equal_a, equal_b)), same_mask_max_errors
    for handle in handles:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    rgb.save(args.output / 'input.png')
    Image.fromarray((mask[0, 0].cpu().numpy() * 255).astype('uint8')).save(args.output / 'mask.png')
    pixels = ((filled[0].permute(1, 2, 0).cpu() + 1) * 127.5).round().clamp(0, 255).byte()
    Image.fromarray(pixels.numpy()).save(args.output / 'completed.png')
    report = dict(
        checkpoint=str(Path(args.checkpoint).resolve()), device=args.device,
        impl=args.impl, feature_shapes=shapes, branch_pair_seconds=seconds,
        full_stage_calls=counts, known_pixels_preserved=True,
        identical_mask_features_close=True, same_mask_max_errors=same_mask_max_errors,
        caller_rng_preserved=True,
        frozen_after_train=True,
        note='Central square test mask, not a smoke prediction or accuracy evaluation.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
