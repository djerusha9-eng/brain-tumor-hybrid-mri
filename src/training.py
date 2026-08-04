from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .data import load_image
from .preprocessing import preprocess_image


class MRISequence:
    """Keras-compatible deterministic sequence with on-demand MRI preprocessing."""

    def __new__(cls, *args, **kwargs):
        import tensorflow as tf

        class _Sequence(tf.keras.utils.Sequence):
            def __init__(self, frame, classes, config, training=False):
                super().__init__()
                self.frame = frame.reset_index(drop=True).copy()
                self.classes = list(classes)
                self.class_to_index = {label: index for index, label in enumerate(classes)}
                self.config = config
                self.training = training
                self.batch_size = int(config["training"]["batch_size"])
                self.seed = int(config["project"]["seed"])
                self.indices = np.arange(len(self.frame))
                self.epoch = 0
                self.on_epoch_end()

            def __len__(self):
                return int(np.ceil(len(self.frame) / self.batch_size))

            def __getitem__(self, batch_index):
                selected = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
                images, targets = [], []
                for row_index in selected:
                    row = self.frame.iloc[row_index]
                    image = load_image(row["path"], self.config["data"]["representative_slice"])
                    image, _ = preprocess_image(
                        image,
                        self.config["preprocessing"],
                        self.config["segmentation"],
                        self.seed,
                    )
                    if self.training:
                        image = self._augment(image, row_index)
                    images.append(image)
                    target = np.zeros(len(self.classes), dtype=np.float32)
                    target[self.class_to_index[row["label"]]] = 1.0
                    targets.append(target)
                return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.float32)

            def _augment(self, image, row_index):
                rng = np.random.default_rng(self.seed + 100000 * self.epoch + int(row_index))
                if rng.random() < 0.5:
                    image = np.flip(image, axis=1)
                rotations = int(rng.integers(0, 4))
                return np.ascontiguousarray(np.rot90(image, rotations, axes=(0, 1)))

            def on_epoch_end(self):
                if self.training:
                    rng = np.random.default_rng(self.seed + self.epoch)
                    rng.shuffle(self.indices)
                self.epoch += 1

        return _Sequence(*args, **kwargs)


def class_weights(frame: pd.DataFrame, classes: list[str]) -> dict[int, float]:
    labels = frame["label"].to_numpy()
    weights = compute_class_weight(class_weight="balanced", classes=np.asarray(classes), y=labels)
    return {index: float(weight) for index, weight in enumerate(weights)}


def fit_model(model, manifest: pd.DataFrame, classes: list[str], config: dict[str, Any]):
    import tensorflow as tf

    train_sequence = MRISequence(manifest[manifest["split"] == "train"], classes, config, training=True)
    validation_sequence = MRISequence(
        manifest[manifest["split"] == "validation"], classes, config, training=False
    )
    model_path = Path(config["artifacts"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(config["training"]["patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=max(2, int(config["training"]["patience"]) // 2)
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    weights = class_weights(manifest[manifest["split"] == "train"], classes)
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=int(config["training"]["epochs"]),
        callbacks=callbacks,
        class_weight=weights,
        verbose=2,
    )
    history_path = Path(config["artifacts"]["history"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    return history

