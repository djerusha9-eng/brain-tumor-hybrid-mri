# Clinical and Methodological Limitations

This implementation has no prospective clinical validation, regulatory clearance, or demonstrated effect on patient outcomes. Public benchmark performance cannot establish safety across hospitals, scanner vendors, acquisition protocols, demographic groups, tumor subtypes, or treatment stages.

The FCM stage is intensity-driven and may respond to artifacts, normal enhancing structures, postoperative change, or acquisition variation. A visually plausible mask is not equivalent to expert tumor delineation. When reference masks exist, segmentation Dice, intersection-over-union, sensitivity, and Hausdorff distance must be reported separately from classification metrics.

The 2-D representative-slice design reduces computational cost but discards volumetric context. Results should not be generalized to full-volume clinical interpretation without a dedicated 3-D evaluation.

SHAP values explain model behavior relative to a selected background distribution. They do not demonstrate pathology, causality, fairness, or clinical correctness. Expert review of selected examples cannot replace a prespecified reader study.

