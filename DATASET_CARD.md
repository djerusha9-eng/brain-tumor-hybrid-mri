# Dataset Card

## Declared sources

The manuscript describes BraTS tumor MRI and IXI non-pathological MRI. Files are not distributed in this repository. Users must follow the access conditions, attribution requirements, and usage rules of each dataset custodian.

## Experimental unit

The patient is the independent experimental unit. Multiple slices or modalities from one patient must remain in the same partition.

## Manifest fields

| Field | Meaning |
|---|---|
| `sample_id` | Unique sample identifier |
| `patient_id` | Stable patient identifier used for splitting |
| `path` | Local image or NIfTI path |
| `label` | Experiment-specific target label |
| `split` | `train`, `validation`, or `test`; may initially be empty |
| `mask_path` | Optional reference segmentation |
| `source` | Dataset or institution identifier |
| `modality` | MRI contrast such as T1CE |

## Label compatibility

BraTS and IXI can support an HGG/LGG/normal experiment when mappings are documented. They do not provide a complete glioma/meningioma/pituitary/normal four-class cohort. A four-class study must cite and document an additional appropriate source.

## Exclusion and quality control

Every exclusion must be recorded before test evaluation with a machine-readable reason. The validator rejects missing files, empty identifiers, unsupported labels, repeated sample identifiers, patient overlap, and duplicate content crossing partitions.

