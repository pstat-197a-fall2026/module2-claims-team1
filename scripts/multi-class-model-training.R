# PRIMARY TASK (b): multi-class classifier
# linear SVM on tf-idf word features from the full page text
# outputs: results/multiclass-model.rds (deployable model)
#           results/multiclass-model-eval.rds (held-out accuracy table)

source('scripts/preprocessing-binary.R')
library(e1071)
library(glmnet)
library(rsample)
library(yardstick)

## PREPROCESSING


load('data/claims-raw.RData')

# parse full page text (takes a few minutes)
claims_text <- claims_raw %>%
  select(.id, text_tmp, mclass) %>%
  parse_pages()
save(claims_text, file = 'data/claims-clean-multiclass.RData')
# next time, skip the parse step and run: load('data/claims-clean-multiclass.RData')

tokens <- tokenize_pages(claims_text)
y_all <- setNames(factor(claims_text$mclass), claims_text$.id)

## HELD-OUT EVALUATION


# same 80/20 split of pages as the preliminary tasks
load('data/claims-clean-example.RData')
set.seed(110122)
partitions <- initial_split(claims_clean, prop = 0.8)
train_ids <- training(partitions)$.id
test_ids <- testing(partitions)$.id

vocab_tr <- fit_vocab(tokens %>% filter(.id %in% train_ids), min_docs = 5)
x_train <- tfidf_matrix(tokens, train_ids, vocab_tr)
x_test <- tfidf_matrix(tokens, test_ids, vocab_tr)
y_train <- y_all[train_ids]
y_test <- y_all[test_ids]

# accuracy, sensitivity, specificity ('Relevant claim content' = positive)
metrics <- function(pred, truth){
  pred_f <- factor(pred, levels = levels(truth))
  tibble(
    accuracy    = accuracy_vec(truth, pred_f),
    sensitivity = sens_vec(truth, pred_f, estimator = 'macro'),
    specificity = spec_vec(truth, pred_f, estimator = 'macro'),
    macro_f1    = f_meas_vec(truth, pred_f, estimator = 'macro')
  )
}

# tune SVM cost with 5-fold cross-validation on training pages
costs <- c(0.1, 0.3, 1, 3)
cv_acc <- sapply(costs, function(C){
  set.seed(1)
  svm(x_train, y_train, kernel = 'linear', cost = C,
      cross = 5, scale = FALSE)$tot.accuracy / 100
})
best_cost <- costs[which.max(cv_acc)]
tibble(cost = costs, cv_acc = cv_acc)

svm_fit <- svm(x_train, y_train, kernel = 'linear', cost = best_cost, scale = FALSE)
svm_pred <- predict(svm_fit, x_test)

# ridge logistic regression on the same features
set.seed(1)
ridge_cv <- cv.glmnet(x_train, y_train, family = 'multinomial', alpha = 0,
                      nfolds = 10, type.measure = 'class')
ridge_pred <- predict(ridge_cv, x_test, s = 'lambda.min', type = 'class')[, 1]

eval_tbl <- bind_rows(
  metrics(svm_pred, y_test) %>% mutate(model = 'Linear SVM (final)', .before = 1),
  metrics(ridge_pred, y_test) %>% mutate(model = 'Ridge logistic', .before = 1)
) %>% mutate(n_test = length(test_ids), cv_accuracy = c(max(cv_acc), 1 - min(ridge_cv$cvm)))
eval_tbl
saveRDS(eval_tbl, 'results/multiclass-model-eval.rds')

## FINAL MODEL: refit on all labeled pages


all_ids <- claims_text$.id
vocab <- fit_vocab(tokens, min_docs = 5)
x_all <- tfidf_matrix(tokens, all_ids, vocab)
final_fit <- svm(x_all, y_all[all_ids], kernel = 'linear',
                 cost = best_cost, scale = FALSE)

# everything needed to score new pages
multiclass_model <- list(vocab = vocab,
                         fit = final_fit,
                         levels = levels(claims_raw$mclass))
saveRDS(multiclass_model, 'results/multiclass-model.rds')