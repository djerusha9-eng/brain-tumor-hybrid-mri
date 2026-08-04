from __future__ import annotations

from typing import Any


FEATURE_DIMENSIONS = {
    "inception_v3": 2048,
    "resnet152_v2": 2048,
    "inception_resnet_v2": 1536,
}


def build_hybrid_model(config: dict[str, Any], classes: list[str]):
    """Build the manuscript-aligned 5,632-dimensional late-fusion network."""
    import tensorflow as tf

    model_config = config["model"]
    height, width = map(int, model_config["image_size"])
    inputs = tf.keras.Input((height, width, 3), name="mri")
    weights = "imagenet" if model_config.get("imagenet_weights", True) else None
    features = []
    for name in model_config["backbones"]:
        backbone, preprocess = _backbone(name, (height, width, 3), weights)
        trainable_layers = int(model_config.get("trainable_backbone_layers", 0))
        backbone.trainable = trainable_layers != 0
        if trainable_layers > 0:
            for layer in backbone.layers[:-trainable_layers]:
                layer.trainable = False
        branch_input = tf.keras.layers.Lambda(
            lambda tensor, fn=preprocess: fn(tensor * 255.0), name=f"{name}_preprocess"
        )(inputs)
        branch_output = backbone(branch_input, training=False)
        features.append(branch_output)

    expected = sum(FEATURE_DIMENSIONS[name] for name in model_config["backbones"])
    fused = tf.keras.layers.Concatenate(name=f"late_fusion_{expected}d")(features)
    for index, units in enumerate(model_config["dense_units"], start=1):
        fused = tf.keras.layers.Dense(
            int(units),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(float(model_config["l2"])),
            name=f"fusion_dense_{index}",
        )(fused)
        fused = tf.keras.layers.BatchNormalization(name=f"fusion_bn_{index}")(fused)
        fused = tf.keras.layers.Dropout(float(model_config["dropout"]), name=f"fusion_dropout_{index}")(fused)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax", name="class_probabilities")(fused)
    model = tf.keras.Model(inputs, outputs, name="inception_resnet_late_fusion")
    model.class_names = list(classes)
    return model


def _backbone(name: str, input_shape: tuple[int, int, int], weights: str | None):
    import tensorflow as tf

    factories = {
        "inception_v3": (
            tf.keras.applications.InceptionV3,
            tf.keras.applications.inception_v3.preprocess_input,
        ),
        "resnet152_v2": (
            tf.keras.applications.ResNet152V2,
            tf.keras.applications.resnet_v2.preprocess_input,
        ),
        "inception_resnet_v2": (
            tf.keras.applications.InceptionResNetV2,
            tf.keras.applications.inception_resnet_v2.preprocess_input,
        ),
    }
    if name not in factories:
        raise ValueError(f"Unsupported backbone {name!r}; choose from {sorted(factories)}")
    factory, preprocess = factories[name]
    return factory(include_top=False, weights=weights, input_shape=input_shape, pooling="avg"), preprocess


def compile_model(model, config: dict[str, Any]):
    import tensorflow as tf

    learning_rate = float(config["training"]["learning_rate"])
    smoothing = float(config["model"].get("label_smoothing", 0.0))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model

