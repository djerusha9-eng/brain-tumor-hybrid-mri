import pytest

from src.models import FEATURE_DIMENSIONS


def test_declared_feature_fusion_dimension_matches_manuscript():
    assert sum(FEATURE_DIMENSIONS.values()) == 5632


def test_unknown_backbone_is_rejected():
    tf = pytest.importorskip("tensorflow")
    from src.models import _backbone

    with pytest.raises(ValueError, match="Unsupported backbone"):
        _backbone("unknown", (299, 299, 3), None)

