# FoSp with full pretrained MAT

This is a local FoSp variant, not an exact reproduction of the published FoSp.
The upstream `MAT-main` directory is left unchanged. Its inference runtime is
vendored in `mat_inpainter/`, with the original license in `mat_inpainter/LICENSE`.

## Separation data flow

Both branches share ONE frozen MAT generator, including all pretrained weights:

```
RGB image + Focus mask -> MAT stage 1 -> MAT stage 2 through 512x512 -> four features
RGB image + all-ones mask -> same full stage 1 -> same full stage 2 -> four features
                               |
                    10 * abs(feature difference)
                               |
                  trainable channel projections -> Domain Fusion -> segmentation
```

Mask 0 means fill; mask 1 means keep. MAT stage 1 actually masks the RGB input,
creates an RGB completion, and feeds it to stage 2. No stage is bypassed, even
for the all-ones branch. The full decoder executes through its 256 and 512 blocks.
FoSp taps intermediate outputs after decoder blocks at:

| Resolution | Channels | FoSp target |
| --- | --- | --- |
| 16 x 16 | 512 | backbone 1/32 |
| 32 x 32 | 512 | backbone 1/16 |
| 64 x 64 | 512 | backbone 1/8 |
| 128 x 128 | 256 | backbone 1/4 |

The final MAT RGB image is available for inspection, but FoSp uses the four
feature maps for segmentation. Therefore the last two decoder blocks execute
without changing the earlier tapped features. This intentionally implements the
requested **full MAT** rather than optimizing away its final RGB computation.

## Checkpoint and commands

Run from this FoSp root, in an environment with the original FoSp/MMCV dependencies:

```
python tools/download_mat.py
python tools/check_full_mat.py --device cuda --impl ref
python tools/train.py config/fosp_config.py
```

The downloader uses a pinned public mirror of `Places_512_FullData.pkl` (661 MB):
https://huggingface.co/Icar/mat_places512_full/tree/476b8ea5b3f82f1784cde1fa0578c710e4a323cd

SHA256: `d960c4e6b3266b6b9fa74ee4458a9482160d54c06d7738696bc9a9e2b34c66dc`.
Official MAT project: https://github.com/fenglinglwb/MAT
This is a scene-inpainting checkpoint, not a smoke-specific checkpoint or a
confirmed copy of the unpublished pretrained weights used in the FoSp paper.

The default checkpoint path is `pretrained/Places_512_FullData.pkl`.
To use another compatible checkpoint:

```
python tools/train.py config/fosp_config.py --cfg-options model.decode_head.mat_checkpoint=/absolute/path/to/Places_512_FullData.pkl
```

The loader uses `G_ema` from a trusted MAT-format `.pkl`, or a plain full-MAT
`state_dict` in `.pth/.pt`. Every parameter and buffer must match (`strict=True`).
Missing or truncated checkpoints fail explicitly; there is no random fallback.
The source checkpoint is required during construction, also before restoring
a complete FoSp training checkpoint.

## Execution details

- MAT is frozen and remains in evaluation mode after parent `model.train()`.
- Both branches use the same latent code, fixed seed and replayed random state.
  This also pairs upstream MAT's functional dropout, which runs even in eval.
  Equal masks therefore produce equal features up to floating-point rounding.
  The caller's RNG is restored.
- MAT runs in FP32 without gradient recording. Focus, feature projections and
  Fusion remain trainable. A hard Focus mask does not backpropagate through MAT.
- Input normalization is reversed on the input device, preserving RGB order.
- The input to MAT is resized to 512x512; its features are resized to backbone
  scales as needed. Parent model placement controls the device; no `.cuda()` is
  called in the new adapter constructor.
- `mat_impl='cuda'` uses upstream fused operators, requiring a compatible CUDA
  compiler/toolchain. `mat_impl='ref'` uses the upstream PyTorch reference
  implementations on CPU or GPU and needs no custom extension compilation:

```
python tools/train.py config/fosp_config.py --cfg-options model.decode_head.mat_impl=ref
```

The kernel implementation setting is shared by MAT adapters in the same process.
Do not instantiate adapters with different `mat_impl` settings in one process.

The config starts with batch size 1 per GPU because full MAT is heavier. Increase
only after measuring memory. This differs from the original batch size 6.
Existing truncated-FoSp checkpoints are not valid full-MAT pretrained weights.
Retrain the segmentation model for this variant; no IoU improvement is assumed.

## Verification

```
python -m pytest tests/test_full_mat.py -q
python tools/check_full_mat.py --device cuda --impl ref
python tools/check_fosp_mat.py --device cuda
```

The first command checks strict loading, freezing, paired randomness, unchanged
caller RNG and input validation with a lightweight stochastic test double.
The second runs real pretrained MAT at 512x512 and checks full stage execution,
four feature shapes, finite outputs, preserved known pixels, and same-mask
feature cancellation within FP32 tolerance. It saves input/mask/completion and a JSON report
under `work_dirs/mat_check/`. The square mask is only a wiring test, not an
evaluation of smoke removal or segmentation accuracy.

`check_fosp_mat.py` runs the complete FoSp forward, backward and inference at
512x512. It checks gradients for the backbone, BCG, channel projections, Fusion
and classification, and verifies MAT remains frozen and retains pretrained
weights after MMSeg initialization. The check disables only the backbone download
and uses synthetic labels; it does not measure IoU. Normal training retains the
MiT-B3 pretrained setting in the config.

Local validation completed on Quadro T1000 (4 GB), PyTorch 2.11, MMCV 1.6.2,
with reference MAT kernels: 6 unit tests passed, real pretrained MAT checks
passed, and FoSp 512x512 forward/backward/inference passed. Reports are saved in
`work_dirs/mat_check/report.json` and `work_dirs/mat_check/fosp_report.json`.
The isolated `.venv-mat` is at the workspace root; original environments are not
modified. MMCV's setup script needed a local packaging-only fix for Python 3.13.
These checks do not establish a complete dataset training run or its accuracy.

Vendored changes from MAT: package-qualified imports, optional feature returns,
reference-kernel selection, and a fix for `noise_mode='none'` in `StyleConv`.
Generator parameter names/shapes and the complete two-stage forward architecture
are retained for strict pretrained loading.
