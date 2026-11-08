import torch

from mmseg.models.decode_heads.base_decode_head import BaseDecodeHead


class TinyHead(BaseDecodeHead):
    def forward(self, inputs):
        return inputs


def test_single_channel_accuracy_thresholds_logits_and_ignores_padding():
    head = TinyHead(in_channels=1, channels=1, num_classes=2, out_channels=1,
                    threshold=0.4, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True))
    logits = torch.tensor([[[[-5., 5., 10.]]]], requires_grad=True)
    label = torch.tensor([[[[0, 1, 255]]]])
    losses = head.losses(logits, label)
    assert losses['acc_seg'].item() == 100
    losses['loss_ce'].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 2] == 0
