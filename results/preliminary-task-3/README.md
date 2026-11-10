# Preliminary task 3

This folder contains the neural-network experiment and all of its generated files. Existing Task 1/2 files and the shared preprocessing and writeups are left unchanged.

From the repository root, run:

```sh
Rscript results/preliminary-task-3/preliminary-task-3.R
```

Or in RStudio, open `module2-claims.Rproj` and run:

```r
source("results/preliminary-task-3/preliminary-task-3.R")
run_task3()
```

The analysis uses installed R packages (`tidyverse`, `tidymodels`, `tidytext`, `textstem`, `nnet`); it does not require TensorFlow. It reads `data/claims-clean-example.RData`, reproduces the seed and outer split used by Tasks 1/2, and compares binary classifiers on the same usable held-out pages. Training-only TF-IDF, 50 principal components, and PC scaling are shared by logistic regression and the network. An inner validation split selects among nine network configurations before test scoring. The network has one sigmoid hidden layer, sigmoid output, direct input-output connections, binary cross-entropy loss, and weight decay, trained by BFGS for at most 1000 iterations (not epochs).

Generated files:

- `word-counts.rds`: token-count cache, checked against the input data checksum.
- `split-ids.rds`: training, validation and held-out page IDs, excluded IDs, and labels.
- `validation-results.csv`: candidate settings, validation metrics, convergence codes.
- `selected-configuration.rds`: configuration selected before held-out evaluation.
- `test-metrics.csv`: final held-out accuracy, sensitivity, specificity and log loss.
- `heldout-predictions.csv` / `.rds`: page-level predictions for both models.
- `task3-models.rds`: fitted network, logistic baseline, feature transformations and label order.
- `task3-summary.md`: findings and a concise methods description for the team writeup.
- `session-info.txt`: R and package versions.
- `run.log`: console output captured during the completed run.

The historical 79% Task 1 figure is context, not a hardcoded result. The script recomputes a logistic baseline without held-out IDF information and reports explicitly whether the selected network matches it. Do not tune further on the same held-out results. Validation and final fits report optimizer convergence, and the paired comparison accounts for using the same evaluation pages.

To inspect saved outputs:

```r
read.csv("results/preliminary-task-3/test-metrics.csv")
readRDS("results/preliminary-task-3/selected-configuration.rds")
models <- readRDS("results/preliminary-task-3/task3-models.rds")
```

All generated outputs are overwritten when the experiment is rerun. No files outside this folder are written by the analysis. This is a preliminary binary experiment; it does not generate the final multiclass model or submission predictions.
