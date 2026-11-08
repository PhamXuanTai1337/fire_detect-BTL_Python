"""Frozen, pretrained, full MAT shared by the two Separation branches."""

import importlib
import inspect
import io
import pickle
from pathlib import Path

import torch
from torch import nn


def _package_source(source):
    # MAT pickles embed the original Python source. Resolve its imports to our
    # vendored runtime, without inserting generic 'networks' into sys.modules.
    return (source.replace("sys.path.insert(0, '../')", '')
            .replace('from torch_utils', 'from mat_inpainter.torch_utils')
            .replace('from networks.', 'from mat_inpainter.networks.')
            .replace('import dnnlib', 'from mat_inpainter import dnnlib'))


def _reconstruct_mat_object(meta):
    from .torch_utils import persistence
    meta = dict(meta)
    meta['module_src'] = _package_source(meta['module_src'])
    return persistence._reconstruct_persistent_obj(meta)


def _torch_load_cpu(source, weights_only=False):
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = weights_only
    return torch.load(source, **kwargs)


class _MATUnpickler(pickle.Unpickler):
    """Load trusted MAT .pkl releases, including GPU-saved storages, on CPU."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: _torch_load_cpu(io.BytesIO(data))
        if module == 'torch_utils.persistence' and name == '_reconstruct_persistent_obj':
            return _reconstruct_mat_object
        if module.split('.')[0] in ('torch_utils', 'networks', 'dnnlib'):
            return getattr(importlib.import_module('mat_inpainter.' + module), name)
        return super().find_class(module, name)


def load_pretrained(generator, checkpoint):
    """Require every parameter/buffer of full MAT; never silently load a subset.

    Supported: trusted official-format .pkl (G_ema), or a plain MAT state_dict
    saved as .pth/.pt. Old FoSp/truncated inpainter checkpoints are rejected.
    """
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            'Full MAT requires a pretrained checkpoint: {}. Run '
            '`python tools/download_mat.py` or set model.decode_head.mat_checkpoint.'
            .format(path))
    if path.suffix.lower() == '.pkl':
        with path.open('rb') as stream:
            payload = _MATUnpickler(stream).load()
        if not isinstance(payload, dict) or 'G_ema' not in payload:
            raise ValueError('MAT .pkl must contain the pretrained G_ema generator.')
        state = payload['G_ema'].state_dict()
    else:
        payload = _torch_load_cpu(str(path), weights_only=True)
        state = payload.get('state_dict', payload)
    generator.load_state_dict(state, strict=True)


class FullMATInpainter(nn.Module):
    """Run both full MAT stages for BOTH masks and return four feature pairs.

    Stage 2 runs through its 512x512 RGB output; features are tapped after its
    16, 32, 64 and 128 decoder blocks (channels: 512, 512, 512, 256).
    No stages or decoder blocks are skipped, including the all-ones branch.
    """

    def __init__(self, checkpoint, seed=240, noise_mode='const', impl='cuda'):
        super().__init__()
        if not checkpoint:
            raise ValueError('mat_checkpoint is required; random frozen MAT is not allowed.')
        if not Path(checkpoint).expanduser().is_file():
            raise FileNotFoundError(
                'Missing MAT checkpoint: {}. Run `python tools/download_mat.py`.'
                .format(checkpoint))
        if noise_mode not in ('const', 'none'):
            raise ValueError('Use const or none noise for comparable Separation features.')
        if impl not in ('cuda', 'ref'):
            raise ValueError('MAT impl must be cuda or ref.')
        from .networks.mat import Generator
        from .torch_utils.ops import bias_act, upfirdn2d
        # Reference kernels allow CPU tests and CUDA use without a C++ compiler.
        # This selection is process-wide within the vendored MAT runtime.
        bias_act.default_impl = impl
        upfirdn2d.default_impl = impl
        self.seed = int(seed)
        self.noise_mode = noise_mode
        self.generator = Generator(
            z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3)
        load_pretrained(self.generator, checkpoint)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent model.train() must not enable MAT training or update w_avg.
        return super().train(False)

    def _branch(self, image, mask, z, label):
        device = image.device
        devices = [device.index] if device.type == 'cuda' else []
        # Upstream MAT uses functional dropout(training=True) even in eval.
        # Replay identical randomness in both branches and restore caller RNG.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(self.seed)
            return self.generator(
                image, mask, z, label, noise_mode=self.noise_mode,
                truncation_psi=1, return_features=True)

    @torch.no_grad()
    def forward(self, image, mask, return_images=False):
        if image.ndim != 4 or image.shape[1:] != (3, 512, 512):
            raise ValueError('Full MAT expects Bx3x512x512 RGB images in [-1, 1].')
        if mask.shape != (image.shape[0], 1, 512, 512):
            raise ValueError('MAT mask must be Bx1x512x512 (0=fill, 1=keep).')
        if image.device != next(self.generator.parameters()).device:
            raise ValueError('Move the FoSp/MAT model to the same device as the images.')
        # Pretrained MAT and its fused operators run in FP32, also under AMP.
        with torch.autocast(device_type=image.device.type, enabled=False) if hasattr(torch, 'autocast') else torch.cuda.amp.autocast(enabled=False):
            image = image.float()
            mask = mask.to(device=image.device, dtype=torch.float32)
            latent_rng = torch.Generator(device=image.device).manual_seed(self.seed)
            z = torch.randn(image.shape[0], 512, device=image.device, generator=latent_rng)
            label = image.new_zeros(image.shape[0], 0)
            filled_image, filled_features = self._branch(image, mask, z, label)
            original_image, original_features = self._branch(
                image, torch.ones_like(mask), z, label)
        if return_images:
            return filled_features, original_features, filled_image, original_image
        return filled_features, original_features
