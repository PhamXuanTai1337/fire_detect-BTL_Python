"""FoSp with one Focus-only panorama pass and shared full-FoSp crop passes."""
import torch
from torch.nn import functional as F

from ..builder import SEGMENTORS
from .encoder_decoder_fosp import EncoderDecoderFOSP
from mmseg.smoke_crops import (CROP_BOXES, crop_targets, extract_crop,
                              merge_crop_probabilities, require_panorama,
                              sample_training_crops, select_crops)


@SEGMENTORS.register_module()
class SelectiveCropFOSP(EncoderDecoderFOSP):
    def __init__(self, crop_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.crop_cfg = dict(mode='selective', threshold=0.5, relative_gap=0.10,
                             train_crops=True, selector_loss_weight=1.0)
        self.crop_cfg.update(crop_cfg or {})
        if self.crop_cfg['mode'] not in ('whole', 'all', 'selective'):
            raise ValueError('crop mode must be whole, all, or selective.')
        if self.crop_cfg['mode'] == 'selective' and not hasattr(self.decode_head, 'selector_head'):
            raise ValueError('Selective mode requires decode_head.crop_selector=True.')
        self.last_crop_debug = []

    def forward_focus(self, img):
        """One backbone/BCG pass. Does not call full decode or MAT."""
        require_panorama(img)
        features = self.extract_feat(img)
        focus = self.decode_head.forward_focus(features)
        if not hasattr(self.decode_head, 'selector_head'):
            raise ValueError('Enable crop_selector to score regions.')
        return self.decode_head.selector_head(features[0], focus[1])

    def encode_decode_full(self, img, img_metas):
        """Explicit full FoSp API; output is logits, including for crop training."""
        return super().encode_decode(img, img_metas)

    def predict_probability(self, img, img_metas, mode=None):
        """Prepared input -> probability on the 512-square working image."""
        require_panorama(img)
        mode = mode or self.crop_cfg['mode']
        self.last_crop_debug = []
        if mode == 'whole':
            return self.encode_decode_full(img, img_metas).sigmoid()
        if mode not in ('all', 'selective'):
            raise ValueError('Unknown crop mode.')
        scores = self.forward_focus(img).sigmoid() if mode == 'selective' else None
        outputs = []
        for batch_index in range(img.shape[0]):
            if scores is None:
                indices, fallback = list(range(9)), False
            else:
                indices, fallback = select_crops(scores[batch_index],
                    self.crop_cfg['threshold'], self.crop_cfg['relative_gap'])
            probabilities = []
            for index in indices:
                crop = extract_crop(img[batch_index:batch_index + 1], index)
                probabilities.append(self.encode_decode_full(crop, [{}]).sigmoid())
            probability, counts = merge_crop_probabilities(probabilities, indices)
            outputs.append(probability)
            self.last_crop_debug.append(dict(
                scores=None if scores is None else scores[batch_index].detach(),
                selected=indices, fallback=fallback, boxes=[CROP_BOXES[i] for i in indices],
                count_map=counts.detach(), uncovered=(counts == 0).detach(),
                img_meta=dict(img_metas[batch_index]) if img_metas else {}))
        return torch.cat(outputs)

    def encode_decode(self, img, img_metas):
        # Legacy callers expect finite logits. Probability-based inference below
        # preserves exact zero outside selected crops, without another sigmoid.
        probability = self.predict_probability(img, img_metas)
        return torch.logit(probability.clamp(1e-7, 1 - 1e-7))

    def inference(self, img, img_meta, rescale):
        probability = self.predict_probability(img, img_meta)
        restored = []
        for i, meta in enumerate(img_meta):
            item = probability[i:i + 1]
            if meta.get('flip', False):
                direction = meta.get('flip_direction', 'horizontal')
                if direction not in ('horizontal', 'vertical'):
                    raise ValueError('Unknown flip direction.')
                item = item.flip((3,) if direction == 'horizontal' else (2,))
            if rescale:
                h, w = meta['img_shape'][:2]
                item = F.interpolate(item[..., :h, :w], meta['ori_shape'][:2],
                                     mode='bilinear', align_corners=False)
            restored.append(item)
        if len({tuple(item.shape) for item in restored}) != 1:
            raise ValueError('Batch inference needs matching original shapes; use batch size 1.')
        return torch.cat(restored)

    def forward_train(self, img, img_metas, gt_semantic_seg):
        require_panorama(img)
        losses = {}
        targets, valid = crop_targets(gt_semantic_seg, self.decode_head.ignore_index)
        if hasattr(self.decode_head, 'selector_head'):
            logits = self.forward_focus(img)
            crop_loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction='none')
            losses['selector.loss_bce'] = (self.crop_cfg['selector_loss_weight'] *
                (crop_loss * valid).sum() / valid.sum().clamp_min(1))
            losses['selector.accuracy'] = (((logits.sigmoid() >= self.crop_cfg['threshold'])
                == targets.bool()) & valid).sum().float() / valid.sum().clamp_min(1)
        if not self.crop_cfg['train_crops']:
            losses.update(super().forward_train(img, img_metas, gt_semantic_seg))
            return losses
        selections = sample_training_crops(targets, valid)
        sample_losses = []
        for b, indices in enumerate(selections):
            for index in indices:
                crop_img = extract_crop(img[b:b + 1], index)
                crop_gt = extract_crop(gt_semantic_seg[b:b + 1], index, is_mask=True)
                meta = dict(img_metas[b], img_shape=(512, 512, 3), pad_shape=(512, 512, 3),
                            crop_box=CROP_BOXES[index], panorama_meta=dict(img_metas[b]))
                sample_losses.append(super().forward_train(crop_img, [meta], crop_gt))
        if sample_losses:
            for key in sample_losses[0]:
                losses[key] = sum(sample[key] for sample in sample_losses) / len(sample_losses)
        else:
            # Keep the same logged keys on every DDP rank, even when this rank
            # sees only ignore labels. Do not run MAT for such images.
            zero = next(p for p in self.parameters() if p.requires_grad).flatten()[0] * 0
            heads = [('decode', self.decode_head)]
            if self.with_auxiliary_head:
                if isinstance(self.auxiliary_head, torch.nn.ModuleList):
                    heads += [(f'aux_{i}', head) for i, head in enumerate(self.auxiliary_head)]
                else:
                    heads.append(('aux', self.auxiliary_head))
            for prefix, head in heads:
                terms = head.loss_decode if isinstance(head.loss_decode, torch.nn.ModuleList) else [head.loss_decode]
                for term in terms:
                    losses[f'{prefix}.{term.loss_name}'] = zero
                losses[f'{prefix}.acc_seg'] = zero.detach()
        return losses
