# MOSAIC — Experiment Summary

## Objective

Evaluate whether sequence-derived features from Limit Order Book (LOB) observations can improve short-horizon mid-price movement classification.

## Dataset

- Dataset: FI-2010
- Subset: NoAuction Z-score
- Training and official test partitions: 9 folds
- Input features: 144
- Sequence length: 100 observations
- Sequence stride: 10 observations
- Target: Fifth prediction horizon
- Classes: DOWN, STATIONARY, UP

## Final Model

HistGradientBoostingClassifier

- max_iter: 150
- learning_rate: 0.08
- max_leaf_nodes: 31
- l2_regularization: 1.0
- random_state: 42

Sequence representation: 576 features, comprising latest values, means, standard deviations and trends.

## Official Test Results

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 59.13% | 2.33 pp |
| Macro F1 | 57.10% | 3.14 pp |
| DOWN F1 | 59.60% | 3.22 pp |
| STATIONARY F1 | 51.80% | 10.95 pp |
| UP F1 | 59.90% | 2.99 pp |

## Main Findings

1. Sequence-summary features provide useful information for LOB movement classification.
2. HistGradientBoosting is the strongest model in the current experiments.
3. STATIONARY is the most variable class across folds.
4. The Transformer did not outperform the sequence-aware gradient-boosting baseline in the current experiments.
5. Results vary across folds, indicating that market conditions and class distributions matter.

## Limitations

- Evaluation uses the FI-2010 NoAuction Z-score benchmark.
- The nine folds are not evidence of generalisation across nine independent instruments or markets.
- No transaction costs, execution simulation or trading profitability analysis has been performed.
- Results do not establish causal relationships.
- Model configuration was informed by prior experimentation on the benchmark's training partitions.

## Conclusion

MOSAIC demonstrates a reproducible sequence-feature-based approach to short-horizon LOB movement classification. The fixed HistGradientBoosting configuration achieves 59.13% mean accuracy and 57.10% mean macro F1 across the nine official test folds.

Further work should investigate class imbalance, probability calibration, temporal robustness and the economic relevance of predictions.
