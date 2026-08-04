# Ensemble benchmark summary

These exploratory benchmarks evaluate the existing artifacts without retraining or tuning. Each dataset was sampled to an equal 50/50 class balance. The 1,000-row runs use the three local components; the 100-row runs add the API-backed rubric component and represent the full runtime ensemble.

## Results across runs

| Dataset | n | Ensemble | Ensemble F1 | Best individual (F1) | F1 delta | Paired bootstrap 95% CI | Conclusion |
|---|---:|---|---:|---|---:|---|---|
| `shahxeebhassan/human_vs_ai_sentences` | 1,000 | local | 0.7533 | TF-IDF (0.8370) | -0.0837 | [-0.1131, -0.0556] | Individual model stronger |
| `shahxeebhassan/human_vs_ai_sentences` | 100 | full | 0.6494 | TF-IDF (0.8350) | -0.1856 | [-0.3294, -0.0553] | Individual model stronger |
| `okemdad/human_vs_ai_master_dataset` | 1,000 | local | 0.6412 | DistilBERT (0.6652) | -0.0240 | [-0.0538, +0.0081] | Inconclusive |
| `okemdad/human_vs_ai_master_dataset` | 100 | full | 0.7250 | DistilBERT (0.7209) | +0.0041 | [-0.0651, +0.0688] | Inconclusive |
| `Yashodhar29/ai-vs-human-small` | 1,000 | local | 0.5276 | DistilBERT (0.6064) | -0.0789 | [-0.1188, -0.0423] | Individual model stronger |
| `Yashodhar29/ai-vs-human-small` | 100 | full | 0.5263 | Rubric (0.6290) | -0.1027 | [-0.2230, +0.0159] | Inconclusive |

Across these external samples, the results do not yet demonstrate a consistent advantage for the ensemble over its strongest individual component.

## Component F1

| Dataset / run | CatBoost | DistilBERT | TF-IDF | Rubric | Local ensemble | Full ensemble |
|---|---:|---:|---:|---:|---:|---:|
| Shahxeebhassan / 1,000 | 0.1128 | 0.7534 | **0.8370** | — | 0.7533 | — |
| Shahxeebhassan / 100 | 0.0727 | 0.7207 | **0.8350** | 0.6024 | 0.7356 | 0.6494 |
| Okemdad / 1,000 | 0.5660 | **0.6652** | 0.6476 | — | 0.6412 | — |
| Okemdad / 100 | 0.6420 | 0.7209 | 0.5909 | 0.6667 | **0.7407** | 0.7250 |
| Yashodhar / 1,000 | 0.2432 | **0.6064** | 0.6019 | — | 0.5276 | — |
| Yashodhar / 100 | 0.2373 | 0.5773 | 0.5128 | **0.6290** | 0.4000 | 0.5263 |

Bold marks the highest observed F1 in each row, including the local ensemble where present; it is not a significance claim.

## Ensemble operating characteristics

| Dataset / run | Precision | Recall | ROC AUC | PR AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Shahxeebhassan / local 1,000 | 0.8475 | 0.6780 | 0.8349 | 0.7770 | 0.1908 | 0.1539 |
| Shahxeebhassan / full 100 | 0.9259 | 0.5000 | 0.8352 | 0.8487 | 0.1955 | 0.1324 |
| Okemdad / local 1,000 | 0.8360 | 0.5200 | 0.7898 | 0.8159 | 0.2081 | 0.1416 |
| Okemdad / full 100 | 0.9667 | 0.5800 | 0.8952 | 0.9139 | 0.1540 | 0.1211 |
| Yashodhar / local 1,000 | 0.7672 | 0.4020 | 0.6793 | 0.6947 | 0.2541 | 0.1622 |
| Yashodhar / full 100 | 0.7692 | 0.4000 | 0.6848 | 0.7395 | 0.2352 | 0.1230 |

The default 0.5 threshold consistently favors precision over recall. Ranking performance (ROC/PR AUC) is often stronger than F1, suggesting that calibration and the decision rule may be contributing to the observed results.

## CatBoost ablation

| Dataset / run | Ensemble F1 | F1 without CatBoost | Change |
|---|---:|---:|---:|
| Shahxeebhassan / local 1,000 | 0.7533 | 0.7748 | +0.0215 |
| Shahxeebhassan / full 100 | 0.6494 | 0.7500 | +0.1006 |
| Okemdad / local 1,000 | 0.6412 | 0.6674 | +0.0262 |
| Okemdad / full 100 | 0.7250 | 0.7500 | +0.0250 |
| Yashodhar / local 1,000 | 0.5276 | 0.6452 | +0.1176 |
| Yashodhar / full 100 | 0.5263 | 0.6602 | +0.1339 |

The results do not suggest that CatBoost probabilities are simply reversed: its ROC AUC varies from 0.42 to 0.77. A more likely interpretation is dataset-dependent generalization combined with probabilities that tend to move the ensemble below the positive-class threshold. Removing CatBoost improves F1 in all six samples, making its weighting a useful area for further evaluation. These ablation gains were not bootstrapped and should be confirmed on held-out resume data.

## Interpretation and noise

- All sources contain general text rather than resumes, so these results are best viewed as an out-of-domain robustness check rather than a measure of product performance.
- Each source exposes only a training split; overlap with component training data is unknown.
- The 100-row full runs have wider intervals and are more sensitive to sampling variation. Comparisons between a 100-row run and its 1,000-row counterpart are descriptive rather than controlled estimates of the rubric's effect.
- F1 depends on the fixed 0.5 threshold and the artificial 50/50 prevalence. Production precision, recall, and calibration may differ.
- Performance varies by dataset, which may reflect differences in source, genre, generator family, and text length.

## Possible next steps

1. Build a human/AI resume holdout at the actual inference unit and preserve generator, source, and author groups across splits.
2. Fit component calibration and ensemble weights on a separate validation set; choose a threshold from the product's false-positive/false-negative costs.
3. Evaluate removing or reducing the CatBoost weight as a promising candidate, then confirm the result with paired bootstrap tests on untouched data.
4. Compare the learned ensemble against simple baselines: best single model, mean probability, and majority vote. Report uncertainty and per-domain slices.

Detailed reports and plots are under [`reports/`](reports/).
