"""Explicit initialization policy and audited weight-only migration."""
import torch


NEW_MODULE_PREFIXES = ('decode_head.selector_head.', 'decode_head.shallow_projection.',
                       'decode_head.shallow_fusion_conv.')


def validate_initialization(cfg):
    if not cfg.get('require_explicit_initialization', False):
        return
    mode = cfg.get('initialization')
    if mode not in ('pretrained', 'finetune', 'resume'):
        raise ValueError('Choose --initialization pretrained|finetune|resume before training.')
    if mode == 'pretrained' and (cfg.get('load_from') or cfg.get('resume_from') or cfg.get('auto_resume')):
        raise ValueError('Pretrained initialization cannot load/resume a FoSp checkpoint.')
    if mode == 'finetune' and (not cfg.get('load_from') or cfg.get('resume_from') or cfg.get('auto_resume')):
        raise ValueError('Finetune requires --load-from and forbids optimizer resume.')
    if mode == 'resume' and (not cfg.get('resume_from') or cfg.get('load_from')):
        raise ValueError('Resume requires --resume-from and forbids --load-from.')


def load_smoke_weights(model, state, allow_new_modules=False):
    if hasattr(model, 'module'):
        model = model.module
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    mismatch = [key for key in current.keys() & state.keys()
                if current[key].shape != state[key].shape]
    forbidden = [key for key in missing if not (
        allow_new_modules and key.startswith(NEW_MODULE_PREFIXES))]
    if forbidden or unexpected or mismatch:
        raise ValueError(f'Incompatible FoSp checkpoint: missing={forbidden}, '
                         f'unexpected={unexpected}, shape_mismatch={mismatch}')
    # Only explicitly allowed new modules retain initialization. Strict loading
    # still validates the entire assembled state, including all frozen MAT.
    model.load_state_dict({**current, **state}, strict=True)
    return dict(missing=missing, unexpected=unexpected, shape_mismatch=mismatch)


def load_smoke_checkpoint(model, filename, allow_new_modules=False):
    payload = torch.load(filename, map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload)
    return load_smoke_weights(model, state, allow_new_modules)
