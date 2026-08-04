from __future__ import annotations

import numpy as np
from scipy.signal import wiener

from .segmentation import candidate_mask


def robust_normalize(image: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Image contains no finite pixels.")
    low, high = np.percentile(image[finite], percentiles)
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(np.nan_to_num(image, nan=low, posinf=high, neginf=low), low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def crop_foreground(image: np.ndarray, threshold: float = 0.02, margin: int = 4) -> np.ndarray:
    points = np.argwhere(image > threshold)
    if not len(points):
        return image
    top, left = np.maximum(points.min(axis=0) - margin, 0)
    bottom, right = np.minimum(points.max(axis=0) + margin + 1, image.shape)
    return image[top:bottom, left:right]


def preprocess_image(image: np.ndarray, preprocessing: dict, segmentation: dict, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    import cv2

    normalized = robust_normalize(image, tuple(preprocessing["percentile_clip"]))
    uint8_image = np.round(normalized * 255).astype(np.uint8)
    median_kernel = int(preprocessing["median_kernel"])
    if median_kernel > 1:
        if median_kernel % 2 == 0:
            raise ValueError("Median kernel size must be odd.")
        uint8_image = cv2.medianBlur(uint8_image, median_kernel)
    filtered = wiener(uint8_image.astype(np.float32), (int(preprocessing["wiener_kernel"]),) * 2)
    filtered = robust_normalize(filtered)
    clahe = cv2.createCLAHE(clipLimit=float(preprocessing["clahe_clip_limit"]), tileGridSize=(8, 8))
    enhanced = clahe.apply(np.round(filtered * 255).astype(np.uint8)).astype(np.float32) / 255.0
    if preprocessing.get("crop_foreground", True):
        enhanced = crop_foreground(enhanced, float(preprocessing["foreground_threshold"]))
    mask = None
    if preprocessing.get("use_fcm_roi", True):
        mask = candidate_mask(
            enhanced,
            clusters=int(segmentation["clusters"]),
            fuzziness=float(segmentation["fuzziness"]),
            max_iterations=int(segmentation["max_iterations"]),
            tolerance=float(segmentation["tolerance"]),
            min_component_pixels=int(segmentation["min_component_pixels"]),
            seed=seed,
        )
        if mask.any():
            enhanced = enhanced * mask.astype(np.float32)
    height, width = map(int, preprocessing["output_size"])
    enhanced = cv2.resize(enhanced, (width, height), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb = np.repeat(enhanced[..., None], 3, axis=-1).astype(np.float32)
    return rgb, mask
