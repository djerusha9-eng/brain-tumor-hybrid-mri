# Reproducibility Protocol

1. Obtain the declared datasets from their custodians.
2. Create a manifest with stable patient identifiers and source metadata.
3. Freeze `config.yaml` before accessing test labels.
4. Run `prepare_data.py` to validate files, calculate hashes, and create patient-disjoint splits.
5. Retain the generated prepared manifest and run metadata with the experiment record.
6. Train using only training data; use validation data for early stopping and checkpoint selection.
7. Evaluate the selected checkpoint once on the test set.
8. Preserve predictions, confidence intervals, figures, package versions, and configuration hashes.
9. Repeat the full experiment with multiple preregistered seeds when estimating training variability.

Deterministic GPU execution depends on the TensorFlow, CUDA, and cuDNN combination. The runtime report records the available environment so residual nondeterminism can be disclosed.

Claims of superiority require identical patient partitions, preprocessing, class definitions, and uncertainty-aware statistical comparisons. Comparing only headline accuracy values from different datasets is not valid.

