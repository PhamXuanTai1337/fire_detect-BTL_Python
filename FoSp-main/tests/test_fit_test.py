import numpy as np
import pytest

from run_fit_test import rasterize_boxes, read_boxes


def test_yolo_class_zero_becomes_foreground_one(tmp_path):
    labels = tmp_path / 'sample.txt'
    labels.write_text('0 0.5 0.5 0.5 0.5\n1 0.1 0.1 0.1 0.1\n')
    boxes = read_boxes(labels, smoke_class=0)
    mask = rasterize_boxes(boxes, size=8)
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:6, 2:6] = 1
    np.testing.assert_array_equal(mask, expected)


def test_boxes_clip_at_image_boundary_and_union():
    mask = rasterize_boxes([(0., 0., 0.5, 0.5), (0.25, 0.25, 0.5, 0.5)], size=8)
    assert mask.sum() == 16
    assert set(np.unique(mask)) == {0, 1}


def test_reject_malformed_yolo_annotation(tmp_path):
    labels = tmp_path / 'sample.txt'
    labels.write_text('0 0.5 0.5 0.2\n')
    with pytest.raises(ValueError, match='expected YOLO'):
        read_boxes(labels)
