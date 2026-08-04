# Hybrid Brain-Tumor MRI Reproducibility Framework

This repository implements the computational pipeline described in *Automated Detection of Brain Tumors in MRI Using Hybrid Machine Learning Techniques*. It combines reproducible MRI preprocessing, fuzzy C-means (FCM) region-of-interest segmentation, three transfer-learning feature extractors, late feature fusion, calibrated evaluation, and SHAP-based interpretation.

The implementation is research software for retrospective experiments. It is not a medical device and must not be used to diagnose, triage, or treat patients.

## Implemented pipeline

1. Validate a metadata manifest and reject duplicate, missing, or cross-split patients.
2. Create a deterministic patient-level train/validation/test partition when one is not supplied.
3. Load 2-D image files or a deterministic representative slice from a NIfTI volume.
4. Apply robust intensity clipping, min-max normalization, median denoising, Wiener filtering, CLAHE enhancement, and brain-region cropping.
5. Estimate an FCM foreground/tumor candidate mask and apply it as an optional region-of-interest gate.
6. Extract 2,048 InceptionV3, 2,048 ResNet152V2, and 1,536 InceptionResNetV2 features.
7. Concatenate the resulting 5,632-dimensional vector and classify it with 1,024- and 512-unit regularized dense layers.
8. Report discrimination, calibration, confidence intervals, confusion matrices, latency, and explainability outputs.

## Dataset scope

MRI data are not redistributed. Obtain access from the original dataset custodians and create `dataset_manifest.csv` using one row per independent sample. The required columns are:

```text
sample_id,patient_id,path,label,split,mask_path,source,modality
```

`split` may be empty. When empty, `prepare_data.py` creates patient-disjoint splits. Relative paths are resolved from the manifest location. The code deliberately stops if a patient occurs in more than one split.

The manuscript currently names BraTS and IXI, which support glioma-grade/healthy experiments but do not themselves provide meningioma and pituitary classes. A four-class experiment therefore requires a separately documented dataset containing those labels. The software never synthesizes missing classes or silently relabels data.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only environment, replace `tensorflow` with the matching `tensorflow-cpu` package if required by the platform.

## Quick start

1. Edit `dataset_manifest.csv` and `config.yaml`.
2. Validate and freeze the split:

```bash
python prepare_data.py --config config.yaml
```

3. Train the hybrid model:

```bash
python train.py --config config.yaml
```

4. Evaluate the untouched test set:

```bash
python evaluate.py --config config.yaml
```

5. Generate SHAP explanations:

```bash
python explain.py --config config.yaml
```

6. Run the complete workflow:

```bash
python reproduce.py --config config.yaml
```

Generated data, splits, checkpoints, predictions, figures, and reports are written below `artifacts/` and are excluded from version control.

## Reproducibility commands

```bash
make validate
make test
make reproduce
```

The experiment report records the configuration hash, manifest hash, split hash, package versions, platform, random seed, model parameter count, and hardware information. Reported manuscript values should be copied only from generated result files; no performance value is hard-coded.

## Experimental integrity

- Splits are formed by `patient_id`, not by image or slice.
- Normalization is applied per image and never uses test-set statistics.
- Model selection uses validation data only.
- The test set is evaluated once after model selection.
- Bootstrap confidence intervals resample patients, not correlated slices.
- Missing classes, invalid paths, duplicate samples, and leakage cause immediate failure.
- Segmentation and classification metrics are reported separately.
- SHAP explanations are descriptive and do not establish clinical causality.

## Repository contents

- `src/`: implementation modules.
- `tests/`: integrity and numerical tests.
- `config.yaml`: complete experiment definition.
- `dataset_manifest.csv`: safe metadata template.
- `prepare_data.py`, `train.py`, `evaluate.py`, `explain.py`, `predict.py`: focused command-line programs.
- `reproduce.py`: end-to-end orchestration.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `REPRODUCIBILITY.md`, and `CLINICAL_LIMITATIONS.md`: reporting documentation.

## Citation

Use the metadata in `CITATION.cff`. Repository version and archival identifiers should be added only after an official release is created.

