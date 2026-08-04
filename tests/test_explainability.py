import numpy as np
import pytest

from src.explainability import _collect_images


class EmptySequence:
    def __len__(self):
        return 0


class TwoBatchSequence:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return np.full((1, 4, 4, 3), index, dtype=np.float32), np.zeros((1, 2))


def test_collect_images_preserves_batches():
    result = _collect_images(TwoBatchSequence())
    assert result.shape == (2, 4, 4, 3)
    assert result[1].mean() == 1


def test_empty_explanation_set_is_rejected():
    with pytest.raises(ValueError, match="No images"):
        _collect_images(EmptySequence())

