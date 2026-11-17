# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .encoder_decoder_fosp import EncoderDecoderFOSP
from .selective_crop_fosp import SelectiveCropFOSP

__all__ = ['BaseSegmentor', 'EncoderDecoderFOSP', 'SelectiveCropFOSP']
