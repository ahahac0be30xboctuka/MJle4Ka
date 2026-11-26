# Binary screening results

[Study overview](../README.md) · [Dataset and protocol](../docs/README.md)

## Model comparison

The table reproduces the metrics in the committed CSV. Recall and F1 are for the positive class, which pools the five antibiotic categories.

| Model | Accuracy | Recall | F1 |
| :--- | ---: | ---: | ---: |
| Random Forest | 0.9565 | **0.9865** | **0.9733** |
| Gradient Boosting | 0.9565 | 0.9820 | 0.9732 |
| k-Nearest Neighbors | 0.9565 | 0.9730 | 0.9730 |
| SVM | 0.9420 | 0.9820 | 0.9646 |
| Logistic Regression | 0.9058 | 0.9730 | 0.9432 |

The top three models share the same reported accuracy. Their F1 scores differ by at most 0.0003 on this split. With 222 positives among 276 test measurements, an always-positive rule would score 0.892 F1 and flag all 54 milk controls. [Saved metrics CSV](model_results.csv)

## Random Forest diagnostics

![Random Forest confusion matrix for the archived binary evaluation](figures/confusion_binary_Random_Forest.png)

*Rows are observed classes and columns are predictions. Class 0 is the milk control and class 1 is the pooled antibiotic group.*

The model detected 219 of 222 positives (98.6% recall) and correctly cleared 45 of 54 controls (83.3% specificity). That leaves three missed positives and nine flagged controls at the recorded decision threshold.

[Saved ROC curve](figures/roc_binary_Random_Forest.png)

## Interpretation

The results show that the sampled current profiles carry signal for the binary labels within this dataset. A screening workflow would need to trade off missed positives against false alarms on independently collected samples and report how that balance changes with concentration. The information needed to establish a detection limit or validate a field decision threshold is outside the archived evaluation.
