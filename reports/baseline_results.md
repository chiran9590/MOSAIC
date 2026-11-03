# MOSAIC — Baseline Experiment

## Dataset
- Dataset: FI-2010
- Subset: NoAuction Z-score
- Fold: 1
- Training observations: 39,512
- Testing observations: 38,397
- Features: 144
- Target: FI-2010 label row 149
- Classes: 1, 2, 3

## Results

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.4692 | 0.5430 |
| Macro F1 | 0.43 | 0.41 |
| Weighted F1 | 0.46 | 0.49 |

## Observations

1. XGBoost achieved higher overall accuracy than Logistic Regression.
2. Logistic Regression achieved a higher macro F1 score.
3. XGBoost had very low recall for class 2 (0.02).
4. Both models show that distinguishing all three classes is challenging.
5. Accuracy alone is insufficient for comparing these models.

## Initial conclusion

XGBoost provides a stronger accuracy baseline, while Logistic Regression provides more balanced class-level performance.

The next experiment will investigate whether a sequence-based Transformer can learn temporal patterns that improve predictive performance.
