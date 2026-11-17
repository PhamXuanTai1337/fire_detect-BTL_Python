"""Binary smoke losses with valid-pixel and per-image reductions."""
from torch import nn
from torch.nn import functional as F

from ..builder import LOSSES


def _binary_inputs(pred, label, weight, ignore_index):
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError('Binary smoke losses require N x 1 x H x W logits.')
    if label.ndim == 4:
        label = label.squeeze(1)
    pred = pred[:, 0].float()
    if pred.shape != label.shape:
        raise ValueError('Logits and labels must have matching spatial shapes.')
    valid = label != ignore_index
    if ((label != 0) & (label != 1) & valid).any():
        raise ValueError('Expected labels 0=background, 1=smoke, or ignore_index.')
    mask = valid.to(pred.dtype)
    if weight is not None:
        if weight.ndim == 4:
            weight = weight.squeeze(1)
        mask = mask * weight.to(pred)
    return pred, (label == 1).to(pred.dtype), mask


@LOSSES.register_module()
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=4.0, loss_weight=0.5,
                 loss_name='loss_weighted_bce'):
        super().__init__()
        if pos_weight <= 0 or loss_weight < 0:
            raise ValueError('pos_weight must be positive; loss_weight nonnegative.')
        self.pos_weight, self.loss_weight = float(pos_weight), float(loss_weight)
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, label, weight=None, ignore_index=255, **kwargs):
        pred, target, mask = _binary_inputs(pred, label, weight, ignore_index)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pred.new_tensor(self.pos_weight), reduction='none')
        return self.loss_weight * (loss * mask).sum() / mask.sum().clamp_min(1)


@LOSSES.register_module()
class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.35, beta=0.65, eps=1e-6,
                 loss_weight=0.5, loss_name='loss_tversky'):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0 or eps <= 0 or loss_weight < 0:
            raise ValueError('Invalid Tversky coefficients, epsilon, or loss weight.')
        self.alpha, self.beta, self.eps = float(alpha), float(beta), float(eps)
        self.loss_weight, self._loss_name = float(loss_weight), loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, label, weight=None, ignore_index=255, **kwargs):
        pred, target, mask = _binary_inputs(pred, label, weight, ignore_index)
        probability = pred.sigmoid()
        dims = (1, 2)
        tp = (probability * target * mask).sum(dims)
        fp = (probability * (1 - target) * mask).sum(dims)
        fn = ((1 - probability) * target * mask).sum(dims)
        loss = 1 - (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps)
        valid_images = (mask.sum(dims) > 0).to(loss.dtype)
        return self.loss_weight * (loss * valid_images).sum() / valid_images.sum().clamp_min(1)
