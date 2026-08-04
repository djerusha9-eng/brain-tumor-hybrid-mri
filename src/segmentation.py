from __future__ import annotations

import numpy as np


def fuzzy_c_means(
    image: np.ndarray,
    clusters: int = 4,
    fuzziness: float = 2.0,
    max_iterations: int = 150,
    tolerance: float = 1e-5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster finite image intensities with deterministic fuzzy C-means."""
    if clusters < 2 or fuzziness <= 1:
        raise ValueError("FCM requires clusters >= 2 and fuzziness > 1.")
    values = np.asarray(image, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Image contains no finite pixels.")
    x = values[finite, None]
    quantiles = np.linspace(0, 100, clusters + 2)[1:-1]
    centers = np.percentile(x[:, 0], quantiles).astype(np.float64)
    if np.unique(centers).size < clusters:
        centers += np.random.default_rng(seed).normal(0, 1e-6, size=clusters)

    power = 2.0 / (fuzziness - 1.0)
    for _ in range(max_iterations):
        distances = np.abs(x - centers[None, :]) + 1e-12
        ratios = distances[:, :, None] / distances[:, None, :]
        membership = 1.0 / np.sum(ratios**power, axis=2)
        weights = membership**fuzziness
        updated = np.sum(weights * x, axis=0) / np.maximum(np.sum(weights, axis=0), 1e-12)
        if np.max(np.abs(updated - centers)) < tolerance:
            centers = updated
            break
        centers = updated

    labels_flat = np.full(values.shape, -1, dtype=np.int16)
    memberships_full = np.zeros((values.size, clusters), dtype=np.float32)
    labels_flat[finite] = np.argmax(membership, axis=1)
    memberships_full[finite] = membership.astype(np.float32)
    labels = labels_flat.reshape(image.shape)
    memberships = memberships_full.reshape(*image.shape, clusters)
    return labels, np.asarray(centers), memberships


def candidate_mask(
    image: np.ndarray,
    clusters: int = 4,
    fuzziness: float = 2.0,
    max_iterations: int = 150,
    tolerance: float = 1e-5,
    min_component_pixels: int = 32,
    seed: int = 42,
) -> np.ndarray:
    import cv2

    labels, centers, _ = fuzzy_c_means(
        image, clusters, fuzziness, max_iterations, tolerance, seed
    )
    target = int(np.argmax(centers))
    mask = (labels == target).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= min_component_pixels:
            clean[components == component] = 1
    return clean.astype(bool)


def dice_score(reference: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-7) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    intersection = np.logical_and(reference, prediction).sum()
    return float((2 * intersection + epsilon) / (reference.sum() + prediction.sum() + epsilon))


def iou_score(reference: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-7) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    intersection = np.logical_and(reference, prediction).sum()
    union = np.logical_or(reference, prediction).sum()
    return float((intersection + epsilon) / (union + epsilon))
