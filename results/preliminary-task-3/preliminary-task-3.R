# Preliminary task 3: binary neural network versus logistic PCR.
# From the repository root:
#   Rscript results/preliminary-task-3/preliminary-task-3.R
# In RStudio: source this file, then call run_task3().
# All generated files stay in results/preliminary-task-3/.

task3_root <- function() {
  starts <- getwd()
  script <- grep('^--file=', commandArgs(FALSE), value = TRUE)
  if (length(script)) starts <- c(starts, dirname(sub('^--file=', '', script[1])))
  for (start in starts) {
    here <- normalizePath(start, mustWork = TRUE)
    repeat {
      if (file.exists(file.path(here, 'module2-claims.Rproj'))) return(here)
      parent <- dirname(here)
      if (identical(parent, here)) break
      here <- parent
    }
  }
  stop('Run from the Module 2 repository or pass its path to run_task3(root = ...).')
}

# This matches the starter's word-tokenization/lemmatization rules, but leaves
# TF-IDF estimation to fit_features(), after splitting the pages.
word_counts <- function(pages) {
  pages |>
    dplyr::ungroup() |>
    dplyr::select(.id, text_clean) |>
    tidytext::unnest_tokens(
      token, text_clean, token = 'words',
      stopwords = stringr::str_remove_all(tidytext::stop_words$word, '[[:punct:]]')
    ) |>
    dplyr::mutate(term = textstem::lemmatize_words(token)) |>
    dplyr::filter(stringr::str_length(term) > 2) |>
    dplyr::count(.id, term, name = 'n')
}

term_matrix <- function(counts, ids, vocabulary) {
  # TF denominator includes all retained word tokens, before vocabulary filtering.
  totals <- counts |>
    dplyr::filter(.id %in% ids) |>
    dplyr::group_by(.id) |>
    dplyr::summarise(total = sum(n), .groups = 'drop')
  rows <- counts |>
    dplyr::filter(.id %in% ids, term %in% vocabulary) |>
    dplyr::left_join(totals, by = '.id')
  out <- matrix(0, length(ids), length(vocabulary),
                dimnames = list(as.character(ids), vocabulary))
  out[cbind(match(rows$.id, ids), match(rows$term, vocabulary))] <- rows$n / rows$total
  out
}

fit_features <- function(counts, ids, k = 50L, min_pages = 20L) {
  freq <- counts |>
    dplyr::filter(.id %in% ids) |>
    dplyr::count(term, name = 'documents') |>
    dplyr::filter(documents >= min_pages, documents < length(ids)) |>
    dplyr::arrange(term)
  if (nrow(freq) <= k || length(ids) <= k) stop('Too few training pages or terms for PCA.')
  idf <- stats::setNames(log(length(ids) / freq$documents), freq$term)
  x <- sweep(term_matrix(counts, ids, names(idf)), 2, idf, '*')
  pca <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = k)
  pcs <- pca$x[, seq_len(k), drop = FALSE]
  # Standardize PC scores so all inputs have comparable scales for the network.
  pc_scale <- apply(pcs, 2, stats::sd)
  if (any(!is.finite(pc_scale) | pc_scale <= 0)) stop('Degenerate principal components.')
  # The unused full-length singular values are small, but x is not needed at inference.
  pca$x <- NULL
  list(idf = idf, pca = pca, pc_scale = pc_scale, k = k,
       min_pages = min_pages)
}

apply_features <- function(counts, ids, features) {
  x <- sweep(term_matrix(counts, ids, names(features$idf)), 2, features$idf, '*')
  pc <- predict(features$pca, newdata = x)[, seq_len(features$k), drop = FALSE]
  sweep(pc, 2, features$pc_scale, '/')
}

fit_network <- function(x, y, size, decay, seed, maxit = 1000L) {
  set.seed(seed)
  nnet::nnet(x = x, y = y, size = size, decay = decay,
             entropy = TRUE, skip = TRUE, rang = 0.1,
             maxit = maxit, MaxNWts = 20000L, trace = FALSE)
}

binary_metrics <- function(y, prob) {
  predicted <- as.integer(prob >= 0.5)
  c(accuracy = mean(predicted == y),
    sensitivity = sum(predicted == 1 & y == 1) / sum(y == 1),
    specificity = sum(predicted == 0 & y == 0) / sum(y == 0),
    log_loss = -mean(y * log(pmax(prob, 1e-15)) +
                    (1 - y) * log(pmax(1 - prob, 1e-15))))
}

run_task3 <- function(root = task3_root()) {
  required <- c('tidyverse', 'tidymodels', 'tidytext', 'textstem', 'nnet')
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop('Install missing R packages: ', paste(missing, collapse = ', '))
  suppressPackageStartupMessages(library(dplyr))
  out <- file.path(root, 'results', 'preliminary-task-3')
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  data_env <- new.env()
  load(file.path(root, 'data', 'claims-clean-example.RData'), envir = data_env)
  pages <- dplyr::ungroup(data_env$claims_clean)
  stopifnot(!anyDuplicated(pages$.id), is.factor(pages$bclass), nlevels(pages$bclass) == 2)
  class_levels <- levels(pages$bclass)

  # Reproduce Tasks 1/2's outer split exactly, before filtering tokenless pages.
  set.seed(110122)
  outer <- rsample::initial_split(pages, prop = 0.8)
  train_ids <- rsample::training(outer)$.id
  test_ids <- rsample::testing(outer)$.id
  message('Tokenizing and lemmatizing text (cached inside the Task 3 folder)...')
  input_md5 <- unname(tools::md5sum(file.path(root, 'data', 'claims-clean-example.RData')))
  cache_path <- file.path(out, 'word-counts.rds')
  cache <- if (file.exists(cache_path)) readRDS(cache_path) else NULL
  if (!is.null(cache) && identical(cache$input_md5, input_md5) &&
      identical(cache$version, 1L)) {
    counts <- cache$counts
  } else {
    counts <- word_counts(pages)
    saveRDS(list(input_md5 = input_md5, version = 1L, counts = counts), cache_path)
  }
  usable <- unique(counts$.id)
  train_ids <- train_ids[train_ids %in% usable]
  test_ids <- test_ids[test_ids %in% usable]
  labels <- function(ids) as.integer(pages$bclass[match(ids, pages$.id)]) - 1L

  # An inner split chooses hyperparameters. The outer test pages are not scored
  # until all choices (including the stopping budget) have been locked.
  train_pages <- pages[match(train_ids, pages$.id), ]
  set.seed(310122)
  inner <- rsample::initial_split(train_pages, prop = 0.8, strata = bclass)
  fit_ids <- rsample::training(inner)$.id
  val_ids <- rsample::testing(inner)$.id
  saveRDS(list(train_ids = train_ids, test_ids = test_ids,
               fit_ids = fit_ids, validation_ids = val_ids,
               excluded_ids = pages$.id[!pages$.id %in% usable],
               input_md5 = input_md5, class_levels = class_levels),
          file.path(out, 'split-ids.rds'))
  message('Fitting training-only TF-IDF and PCA for validation...')
  inner_features <- fit_features(counts, fit_ids)
  x_fit <- apply_features(counts, fit_ids, inner_features)
  x_val <- apply_features(counts, val_ids, inner_features)
  y_fit <- labels(fit_ids)
  y_val <- labels(val_ids)

  grid <- expand.grid(size = c(8L, 16L, 32L), decay = c(0.1, 1, 10))
  grid$seed <- 3100L + seq_len(nrow(grid))
  grid$maxit <- 1000L
  grid$accuracy <- grid$log_loss <- NA_real_
  grid$convergence <- NA_integer_
  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    message(sprintf('Validation candidate %d/%d: %d hidden units; decay %g',
                    i, nrow(grid), g$size, g$decay))
    net <- fit_network(x_fit, y_fit, g$size, g$decay, g$seed, g$maxit)
    prob <- as.numeric(predict(net, x_val, type = 'raw'))
    score <- binary_metrics(y_val, prob)
    grid$accuracy[i] <- score['accuracy']
    grid$log_loss[i] <- score['log_loss']
    grid$convergence[i] <- net$convergence
    utils::write.csv(grid, file.path(out, 'validation-results.csv'), row.names = FALSE)
  }
  # Primary selection criterion: accuracy; ties: log loss, then fewer units.
  best <- grid[order(-grid$accuracy, grid$log_loss, grid$size), ][1, ]
  saveRDS(best, file.path(out, 'selected-configuration.rds'))

  message('Refitting selected network and logistic baseline on the full training set...')
  features <- fit_features(counts, train_ids)
  x_train <- apply_features(counts, train_ids, features)
  x_test <- apply_features(counts, test_ids, features)
  y_train <- labels(train_ids)
  y_test <- labels(test_ids)
  net <- fit_network(x_train, y_train, best$size, best$decay, best$seed, best$maxit)
  baseline <- stats::glm(y ~ ., data = data.frame(y = y_train, x_train),
                         family = stats::binomial())
  # Avoid serializing this function's entire workspace through formula environments.
  environment(baseline$terms) <- baseenv()
  environment(baseline$formula) <- baseenv()
  attr(baseline$model, 'terms') <- baseline$terms
  p_net <- as.numeric(predict(net, x_test, type = 'raw'))
  p_baseline <- as.numeric(predict(baseline, newdata = data.frame(x_test), type = 'response'))
  net_score <- binary_metrics(y_test, p_net)
  baseline_score <- binary_metrics(y_test, p_baseline)
  metrics <- data.frame(model = c('Logistic PCR (50 PCs)', 'Neural network'),
                        rbind(baseline_score, net_score), row.names = NULL)
  utils::write.csv(metrics, file.path(out, 'test-metrics.csv'), row.names = FALSE)
  predictions <- data.frame(.id = test_ids,
                            truth = factor(class_levels[y_test + 1L], levels = class_levels),
                            logistic_probability = p_baseline,
                            neural_probability = p_net,
                            logistic_prediction = class_levels[(p_baseline >= 0.5) + 1L],
                            neural_prediction = class_levels[(p_net >= 0.5) + 1L])
  saveRDS(predictions, file.path(out, 'heldout-predictions.rds'))
  utils::write.csv(predictions, file.path(out, 'heldout-predictions.csv'), row.names = FALSE)
  saveRDS(list(network = net, baseline = baseline, features = features,
               class_levels = class_levels, configuration = best,
               input_md5 = input_md5), file.path(out, 'task3-models.rds'))

  net_correct <- (p_net >= 0.5) == y_test
  base_correct <- (p_baseline >= 0.5) == y_test
  gain <- sum(net_correct & !base_correct)
  loss <- sum(!net_correct & base_correct)
  paired_p <- if (gain + loss > 0) stats::binom.test(gain, gain + loss)$p.value else 1
  ci <- stats::binom.test(sum(net_correct), length(y_test))$conf.int
  met <- net_score['accuracy'] >= baseline_score['accuracy']
  pct <- function(x) sprintf('%.1f%%', 100 * x)
  report <- c(
    '# Preliminary task 3: neural network', '',
    sprintf('On %d held-out pages, the neural network achieved **%s accuracy**, compared with **%s** for logistic principal component regression.',
            length(test_ids), pct(net_score['accuracy']), pct(baseline_score['accuracy'])), '',
    if (met) 'The selected network matched or exceeded the recomputed logistic baseline on this split. The historical Task 1 result is a separate comparison, described below.' else
      'The selected network did not meet the task requirement on this split. This result should be reported as incomplete; further experiments need fresh evaluation data or nested resampling.', '',
    '## Method', '',
    sprintf('We used the supplied paragraph-cleaned text, word tokenization, stop-word removal and lemmatization. Terms occurring in at least 20 training pages were represented by TF-IDF and reduced to 50 principal components. Vocabulary, IDF weights, PCA and input scales were estimated using training pages only. The logistic baseline used the same representation and pages.'), '',
    sprintf('The network had 50 inputs, one hidden layer of %d sigmoid units, a sigmoid binary output, and direct input-to-output connections. It minimized binary cross-entropy plus an L2 weight penalty (decay %g) using nnet\'s BFGS optimizer, with a maximum of %d iterations. These are optimizer iterations, not mini-batch epochs. Nine configurations varied hidden-layer size (8, 16, 32) and decay (0.1, 1, 10).',
            best$size, best$decay, best$maxit), '',
    sprintf('The same seed and outer 80/20 splitting procedure as the Task 1–2 scripts were used, excluding pages with no usable tokens. Package versions and usable-page counts can differ from the historical writeup; split IDs are saved for reproducibility. A stratified 80/20 split inside the training set selected the configuration by validation accuracy, with log loss breaking ties. Selected validation accuracy was %s. The final model was refitted on all %d usable training pages; held-out test pages were evaluated afterward.',
            pct(best$accuracy), length(train_ids)), '',
    '## Results', '',
    '| Model | Accuracy | Sensitivity | Specificity |',
    '|---|---:|---:|---:|',
    sprintf('| Logistic PCR | %s | %s | %s |', pct(baseline_score['accuracy']),
            pct(baseline_score['sensitivity']), pct(baseline_score['specificity'])),
    sprintf('| Neural network | %s | %s | %s |', pct(net_score['accuracy']),
            pct(net_score['sensitivity']), pct(net_score['specificity'])), '',
    sprintf('Sensitivity treats `%s` as the positive class; specificity refers to `%s`. The network accuracy has a 95%% exact binomial interval of %s–%s. A paired exact test of prediction disagreements gives p = %.3f (%d neural-only correct; %d logistic-only correct). A numerical gain alone does not establish a reliable advantage.',
            class_levels[2], class_levels[1], pct(ci[1]), pct(ci[2]), paired_p, gain, loss), '',
    sprintf('Final optimizer convergence code: %d (0 means converged; 1 means the iteration budget was reached).', net$convergence), '',
    '## Interpretation and limits', '',
    'The historical Task 1 writeup reports about 79% accuracy on 391 pages. The baseline here is recomputed with the currently installed packages, training-only IDF estimates and the usable pages identified in this run; it can therefore differ from that historical result. Exact reproduction of the historical comparison requires the original split IDs and environment. The historical matched-page header model reported 78.5%. These results concern usable paragraph-text pages, not the 929 unlabeled submission pages. No final multiclass model or submission predictions were produced.', '',
    'This is one split, and neural-network initialization can affect accuracy. The cached data, split IDs, validation scores, model objects, predictions and session information are saved alongside this script. The cleaned example data are treated as fixed input; no new HTML scraping was performed.'
  )
  writeLines(report, file.path(out, 'task3-summary.md'))
  capture.output(utils::sessionInfo(), file = file.path(out, 'session-info.txt'))
  print(metrics)
  message('Finished. Outputs: ', out)
  invisible(list(metrics = metrics, selected = best, task_requirement_met = unname(met)))
}

if (sys.nframe() == 0L) run_task3()
