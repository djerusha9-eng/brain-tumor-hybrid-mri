# Model Card

## Model summary

The model is a late-fusion MRI classifier. InceptionV3, ResNet152V2, and InceptionResNetV2 produce pooled feature vectors of 2,048, 2,048, and 1,536 elements. Their concatenation is processed by two regularized dense layers and a softmax output. Optional FCM masking focuses the input on a high-intensity candidate region.

## Intended use

The model is intended for controlled retrospective reproducibility experiments and methodological comparison. Outputs may support research into computer-aided MRI analysis.

## Out-of-scope use

The model is not validated for clinical diagnosis, screening, treatment selection, prognosis, autonomous decision-making, or deployment on scanners and populations not represented in the evaluation data.

## Inputs and outputs

- Input: a 2-D MRI image or NIfTI volume with declared modality and patient identifier.
- Output: class probabilities over labels configured for the experiment.
- Optional output: FCM mask, calibrated probabilities, and SHAP feature attributions.

## Evaluation requirements

Results must be reported on a patient-disjoint test set. Required outputs include class counts, confusion matrix, macro-averaged metrics, uncertainty intervals, calibration, and per-source performance. Performance values are generated during execution and are intentionally absent from this card.

## Known limitations

Dataset shift, scanner variation, inconsistent acquisition parameters, 2-D slice selection, weak anatomical specificity of intensity clustering, class imbalance, and dependence on public retrospective datasets can reduce generalization. SHAP attribution does not prove biological or causal relevance.

