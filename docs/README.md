# Dataset and evaluation protocol

[Study overview](../README.md) · [Archived results](../reports/report.md)

## Measurements

The project includes exported instrument measurements in `data/raw/` and a tabular version in `data/processed/processed_data.csv`. Each processed row represents a voltammogram. Its 521 input columns correspond to voltage positions, with current in amperes as the measured value.

| Recorded category | Measurements |
| :--- | ---: |
| Milk control | 270 |
| Cefazoline | 225 |
| Ceftiofur | 225 |
| Penicillin | 225 |
| Streptomycin | 207 |
| Tetracycline | 225 |
| **Total** | **1,377** |

The loader uses the voltage grid of the first eligible trace and reads the category and concentration from the raw filename. The preprocessing step drops incomplete rows. The packaged table records the curve, category and concentration. Batch, sample and instrument identifiers, concentration units and an acquisition protocol are undocumented in the repository. [Raw data](../data/raw/) · [Loader](../src/data/load_data.py) · [Preprocessing](../src/data/process_data.py)

## Evaluation protocol

1. **Define the target.** Milk control is class 0. The five antibiotic categories are pooled into class 1, accounting for 1,107 of 1,377 measurements.
2. **Split by measurement.** The binary CLI defaults to an 80/20 stratified split with seed 42. It drops both the category and concentration columns from the features.
3. **Compare classifiers.** Logistic regression, k-nearest neighbors, Random Forest, SVM and gradient boosting use the current vectors. The final fit uses a scaler fitted on training rows. If `--tune` is used, the current implementation scales the training partition before grid-search folds are formed.
4. **Report the saved metrics.** Accuracy, recall and F1 are computed on the held-out rows. Recall and F1 treat the antibiotic group as positive. The CLI selects the highest test-set F1 for its saved diagnostic plots.

This split estimates discrimination among rows in the packaged dataset and also serves to rank the models. It leaves generalization to new physical samples, acquisition batches and concentrations unmeasured. For an independent evaluation, identify the physical sample or acquisition batch, keep those groups intact across folds, fit preprocessing inside each fold and reserve separate groups for final evaluation. [CLI implementation](../main.py) · [Model fitting and scoring](../src/models/train.py)

## Running from the repository root

See the [project README](../README.md#run-the-analysis) for Python dependencies.

| Command | Input and output |
| :--- | :--- |
| `python main.py collect` | Reads the instrument CSVs under `data/raw/` and writes the processed feature table. If the included default archive exists, the command extracts it first. |
| `python main.py analyze` | Reads the processed table and writes class counts and example traces. |
| `python main.py train_binary` | Fits five binary classifiers and writes a metrics CSV, a confusion matrix and a ROC curve for the model with the highest F1. |
| `python main.py train_multiclass` | Runs the separate category prediction path. The archived results page reports the binary run. |

Run from a separate checkout to preserve the committed outputs for comparison. Training commands write to the same metrics path.
