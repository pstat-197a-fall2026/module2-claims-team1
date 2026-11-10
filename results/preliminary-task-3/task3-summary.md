# Preliminary task 3: neural network

On 397 held-out pages, the neural network achieved **78.8% accuracy**, compared with **77.1%** for logistic principal component regression.

The selected network matched or exceeded the recomputed logistic baseline on this split. The historical Task 1 result is a separate comparison, described below.

## Method

We used the supplied paragraph-cleaned text, word tokenization, stop-word removal and lemmatization. Terms occurring in at least 20 training pages were represented by TF-IDF and reduced to 50 principal components. Vocabulary, IDF weights, PCA and input scales were estimated using training pages only. The logistic baseline used the same representation and pages.

The network had 50 inputs, one hidden layer of 8 sigmoid units, a sigmoid binary output, and direct input-to-output connections. It minimized binary cross-entropy plus an L2 weight penalty (decay 1) using nnet's BFGS optimizer, with a maximum of 1000 iterations. These are optimizer iterations, not mini-batch epochs. Nine configurations varied hidden-layer size (8, 16, 32) and decay (0.1, 1, 10).

The same seed and outer 80/20 splitting procedure as the Task 1–2 scripts were used, excluding pages with no usable tokens. Package versions and usable-page counts can differ from the historical writeup; split IDs are saved for reproducibility. A stratified 80/20 split inside the training set selected the configuration by validation accuracy, with log loss breaking ties. Selected validation accuracy was 80.7%. The final model was refitted on all 1574 usable training pages; held-out test pages were evaluated afterward.

## Results

| Model | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|
| Logistic PCR | 77.1% | 77.2% | 76.9% |
| Neural network | 78.8% | 81.1% | 75.7% |

Sensitivity treats `Relevant claim content` as the positive class; specificity refers to `N/A: No relevant content.`. The network accuracy has a 95% exact binomial interval of 74.5%–82.8%. A paired exact test of prediction disagreements gives p = 0.382 (27 neural-only correct; 20 logistic-only correct). A numerical gain alone does not establish a reliable advantage.

Final optimizer convergence code: 0 (0 means converged; 1 means the iteration budget was reached).

## Interpretation and limits

The historical Task 1 writeup reports about 79% accuracy on 391 pages. The baseline here is recomputed with the currently installed packages, training-only IDF estimates and the usable pages identified in this run; it can therefore differ from that historical result. Exact reproduction of the historical comparison requires the original split IDs and environment. The historical matched-page header model reported 78.5%. These results concern usable paragraph-text pages, not the 929 unlabeled submission pages. No final multiclass model or submission predictions were produced.

This is one split, and neural-network initialization can affect accuracy. The cached data, split IDs, validation scores, model objects, predictions and session information are saved alongside this script. The cleaned example data are treated as fixed input; no new HTML scraping was performed.
