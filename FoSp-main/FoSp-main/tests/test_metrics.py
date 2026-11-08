import numpy as np
import pytest

from mmseg.core.evaluation.metrics import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.datasets import SmokeDataset


def test_two_row_mask_and_streaming_agree():
    truth = np.array([[0, 1, 1], [0, 0, 255]], dtype=np.uint8)
    pred = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    stats = intersect_and_union(pred, truth, 2, 255)
    direct = eval_metrics([pred], [truth], 2, 255, ['mIoU', 'mFscore'])
    streaming = pre_eval_to_metrics([stats], ['mIoU', 'mFscore'])
    legacy = eval_metrics([([pred], [pred])], [truth], 2, 255, ['mIoU', 'mFscore'])
    assert direct['IoU'][1] == pytest.approx(1 / 3)
    assert direct['IoU'][0] == pytest.approx(1 / 2)
    assert direct['Fscore'][1] == pytest.approx(1 / 2)
    for key in direct:
        np.testing.assert_allclose(direct[key], streaming[key])
        np.testing.assert_allclose(direct[key], legacy[key])
    dataset = SmokeDataset.__new__(SmokeDataset)
    scores = dataset.evaluate([stats], metric=['mIoU', 'mFscore'], logger='silent')
    assert scores['mIoU'] == pytest.approx(5 / 12)
    assert scores['IoU.smoke'] == pytest.approx(1 / 3)


def test_disjoint_foreground_has_zero_fscore():
    truth = np.array([[1, 0]], dtype=np.uint8)
    pred = np.array([[0, 1]], dtype=np.uint8)
    result = eval_metrics([pred], [truth], 2, 255, ['mFscore'])
    np.testing.assert_array_equal(result['Fscore'], [0, 0])


def test_legacy_smoke_metric_pre_eval():
    truth = np.array([[0, 1]], dtype=np.uint8)
    stats = intersect_and_union(truth, truth, 2, 255)
    result = pre_eval_to_metrics([stats], ['SmokeSeg'])
    np.testing.assert_array_equal(result['mIoU_by_image'], [1, 1])
