"""Checkpoint and preprocessing helpers shared by CLI prediction/evaluation."""
from pathlib import Path
import tempfile

import torch
from mmcv import Config

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from mmseg.smoke_checkpoint import load_smoke_weights


def load_for_inference(config, checkpoint, device):
    cfg = Config.fromfile(str(config)) if not isinstance(config, Config) else config.copy()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = payload['state_dict']
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    prefix = 'decode_head.G.generator.'
    mat_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not mat_state:
        raise ValueError('Checkpoint must contain complete MAT weights.')
    cfg.model.pretrained = None
    cfg.model.backbone.pop('init_cfg', None)
    cfg.model.decode_head.mat_impl = 'ref'
    with tempfile.TemporaryDirectory(prefix='fosp_mat_') as temporary:
        path = Path(temporary) / 'mat.pth'
        torch.save(mat_state, path)
        cfg.model.decode_head.mat_checkpoint = str(path)
        model = build_segmentor(cfg.model)
    load_smoke_weights(model, state, allow_new_modules=False)
    model.to(device).eval()
    model.cfg = cfg
    return model, cfg


def prepare_image(config, filename, device, split='test'):
    """Run the exact configured evaluation pipeline, including CLAHE once."""
    pipeline = Compose(config.data[split].pipeline)
    data = pipeline(dict(img_info=dict(filename=str(filename)), img_prefix=None))
    if len(data['img']) != 1:
        raise ValueError('CLI probability/debug export expects one evaluation augmentation.')
    image = data['img'][0].unsqueeze(0).to(device)
    return image, [data['img_metas'][0].data]


def restore_count_map(debug, meta):
    count = debug['count_map'].float()
    if meta.get('flip', False):
        count = count.flip((3,) if meta['flip_direction'] == 'horizontal' else (2,))
    h, w = meta['img_shape'][:2]
    return torch.nn.functional.interpolate(count[..., :h, :w], meta['ori_shape'][:2],
                                           mode='nearest')[0, 0]
