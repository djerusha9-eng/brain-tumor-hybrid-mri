import numpy as np

from src.segmentation import candidate_mask, dice_score, fuzzy_c_means, iou_score


def test_fcm_separates_two_intensity_regions():
    image = np.zeros((40, 40), dtype=np.float32)
    image[12:28, 12:28] = 1.0
    labels, centers, membership = fuzzy_c_means(image, clusters=2)
    assert labels.shape == image.shape
    assert membership.shape == (40, 40, 2)
    assert np.allclose(membership.sum(axis=-1), 1, atol=1e-5)
    assert centers.max() > centers.min()


def test_candidate_mask_and_overlap_metrics():
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:44, 20:44] = 1.0
    mask = candidate_mask(image, clusters=2, min_component_pixels=10)
    reference = image.astype(bool)
    assert dice_score(reference, mask) > 0.95
    assert iou_score(reference, mask) > 0.90

