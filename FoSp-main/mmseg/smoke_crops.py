"""Fixed panorama crop geometry, selection, supervision and probability merge."""
import torch
from torch import nn
from torch.nn import functional as F


# (x1, y1, x2, y2), row-major, on the padded 512-square working image.
CROP_BOXES = tuple((x, y, x + 256, y + 256)
                   for y in (0, 128, 256) for x in (0, 128, 256))


def require_panorama(tensor):
    if tensor.ndim != 4 or tensor.shape[-2:] != (512, 512):
        raise ValueError('Nine-crop input must be NCHW, letterboxed to 512 x 512.')


def extract_crop(tensor, index, is_mask=False):
    require_panorama(tensor)
    x1, y1, x2, y2 = CROP_BOXES[index]
    crop = tensor[..., y1:y2, x1:x2]
    if is_mask:
        return F.interpolate(crop.float(), (512, 512), mode='nearest').to(tensor.dtype)
    return F.interpolate(crop, (512, 512), mode='bilinear', align_corners=False)


def crop_targets(mask, ignore_index=255):
    require_panorama(mask)
    if mask.shape[1] != 1:
        raise ValueError('Crop labels require a single mask channel.')
    if ((mask != 0) & (mask != 1) & (mask != ignore_index)).any():
        raise ValueError('Expected mask labels 0, 1, 255.')
    labels, valid = [], []
    for x1, y1, x2, y2 in CROP_BOXES:
        region = mask[..., y1:y2, x1:x2]
        labels.append((region == 1).flatten(1).any(1))
        valid.append((region != ignore_index).flatten(1).any(1))
    return torch.stack(labels, 1).float(), torch.stack(valid, 1)


def sample_training_crops(labels, valid):
    """One positive and one negative per image, independent of predictions.

    If a class is absent, sample another valid crop (distinct when possible).
    An all-ignore image has no segmentation crops.
    """
    choices = []
    for row, keep in zip(labels, valid):
        selected = []
        for positive in (True, False):
            pool = torch.where(keep & (row.bool() == positive))[0]
            if not len(pool):
                pool = torch.where(keep)[0]
            if not len(pool):
                continue
            unused = pool[torch.tensor([int(i) not in selected for i in pool],
                                       device=pool.device, dtype=torch.bool)]
            if len(unused):
                pool = unused
            selected.append(int(pool[torch.randint(len(pool), (1,), device=pool.device)]))
        choices.append(selected)
    return choices


def select_crops(scores, threshold=0.5, relative_gap=0.10, eps=1e-8):
    if scores.ndim != 1 or scores.numel() != 9:
        raise ValueError('Expected nine independent crop scores.')
    if not 0 <= threshold <= 1 or not 0 <= relative_gap <= 1 or eps <= 0:
        raise ValueError('Invalid crop threshold or relative fallback gap.')
    values = scores.detach().float().cpu().tolist()
    if not all(0 <= value <= 1 for value in values):
        raise ValueError('Crop scores must be finite probabilities in [0, 1].')
    selected = [i for i, value in enumerate(values) if value >= threshold]
    if selected:
        return selected, False
    # Index is the explicit secondary key for reproducible ties.
    order = sorted(range(9), key=lambda i: (-values[i], i))
    first, second = order[:2]
    selected = [first]
    gap = (values[first] - values[second]) / max(values[first], eps)
    if gap <= relative_gap + 1e-7:
        selected.append(second)
    return selected, True


def merge_crop_probabilities(probabilities, indices):
    """Arithmetic mean of probabilities; uncovered pixels are exactly zero."""
    if not indices or len(indices) != len(probabilities) or len(set(indices)) != len(indices):
        raise ValueError('Provide one probability per distinct selected crop.')
    template = probabilities[0]
    sums = template.new_zeros((template.shape[0], 1, 512, 512), dtype=torch.float32)
    counts = torch.zeros_like(sums)
    for probability, index in zip(probabilities, indices):
        if index not in range(9) or probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError('Invalid crop index or probability shape.')
        if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError('Merge accepts probabilities, not logits.')
        x1, y1, x2, y2 = CROP_BOXES[index]
        local = F.interpolate(probability.float(), (256, 256), mode='bilinear', align_corners=False)
        sums[..., y1:y2, x1:x2] += local
        counts[..., y1:y2, x1:x2] += 1
    return sums / counts.clamp_min(1), counts


class CropSelectorHead(nn.Module):
    """Shared head over shallow features + Focus; max pooling retains tiny cues."""
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.projection = nn.Sequential(nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),
                                        nn.ReLU(inplace=True))
        self.score = nn.Linear(2 * hidden_channels, 1)

    def forward(self, shallow, focus_logits):
        if shallow.shape[-2:] != (128, 128):
            raise ValueError('Selector requires stride-4 features of the 512-square panorama.')
        focus = F.interpolate(focus_logits, shallow.shape[-2:], mode='bilinear', align_corners=False)
        feature = self.projection(torch.cat([shallow, focus.sigmoid()], 1))
        logits = []
        for x1, y1, x2, y2 in CROP_BOXES:
            region = feature[..., y1 // 4:y2 // 4, x1 // 4:x2 // 4]
            pooled = torch.cat([region.mean((2, 3)), region.amax((2, 3))], 1)
            logits.append(self.score(pooled).squeeze(1))
        return torch.stack(logits, 1)
