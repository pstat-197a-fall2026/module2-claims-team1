source('scripts/preprocessing.R')
library(tidymodels)

load('data/claims-clean-example.RData')

# split by page id so the same pages can be reused for every comparison
set.seed(110122)
partitions <- initial_split(claims_clean, prop = 0.8)
train_ids <- training(partitions)$.id
test_ids <- testing(partitions)$.id

# word tokenization, one row per page, tf-idf columns (slow, lemmatizing)
wide <- nlp_fn(claims_clean)

# save so you don't have to rerun nlp_fn next time
save(wide, file = 'data/claims-wide.RData')
# next time, skip the nlp_fn step above and run: load('data/claims-wide.RData')

train_df <- wide %>% filter(.id %in% train_ids)
test_df <- wide %>% filter(.id %in% test_ids)

# vocabulary filter: keep words that appear in at least min_pages training pages (for less computations)
min_pages <- 20
word_cols <- setdiff(names(train_df), c('.id', 'bclass'))
page_counts <- colSums(train_df[, word_cols] > 0)
keep_words <- word_cols[page_counts >= min_pages]
length(keep_words)  # how many words are left

x_train <- train_df %>% select(all_of(keep_words)) %>% as.matrix()
x_test <- test_df %>% select(all_of(keep_words)) %>% as.matrix()
y_train <- train_df$bclass
y_test <- test_df$bclass

# number of principal components (fixed so results are comparable)
k <- 50

# PCA on training data only, keeping just the top k components
pca <- prcomp(x_train, center = TRUE, scale. = FALSE, rank. = k)
pc_train <- pca$x
pc_test <- predict(pca, x_test)

# logistic regression on the k components
d_train <- data.frame(y = y_train, pc_train)
fit <- glm(y ~ ., data = d_train, family = binomial)

# accuracy on held-out test pages (this is the number we report)
p <- predict(fit, newdata = data.frame(pc_test), type = 'response')
pred <- factor(levels(y_train)[(p > 0.5) + 1], levels = levels(y_train))
mean(pred == y_test)

acc_para <- mean(pred == y_test)


# training accuracy
p_train <- predict(fit, type = 'response')
pred_train <- factor(levels(y_train)[(p_train > 0.5) + 1], levels = levels(y_train))
mean(pred_train == y_train)

# how many fitted probabilities are essentially 0 or 1 (as per the glm warning message)
sum(p_train < 1e-8 | p_train > 1 - 1e-8)









## TASK 1: PARAGRAPHS + HEADERS
################################

# same as parse_fn, but also pulls header tags
parse_fn_headers <- function(.html){
  tryCatch({
    read_html(.html) %>%
      html_elements('p, h1, h2, h3, h4, h5, h6') %>%
      html_text2() %>%
      str_c(collapse = ' ') %>%
      rm_url() %>%
      rm_email() %>%
      str_remove_all('\'') %>%
      str_replace_all(paste(c('\n', 
                              '[[:punct:]]', 
                              'nbsp', 
                              '[[:digit:]]', 
                              '[[:symbol:]]'),
                            collapse = '|'), ' ') %>%
      str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
      tolower() %>%
      str_replace_all("\\s+", " ")
  }, error = function(e) NA_character_)
}

# same as parse_data, but calls the header version
parse_data_headers <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_headers(text_tmp)) %>%
    unnest(text_clean) %>%
    filter(!is.na(text_clean))
  return(out)
}

# parse raw html with headers included (slow)
load('data/claims-raw.RData')
claims_clean_headers <- claims_raw %>% parse_data_headers()
save(claims_clean_headers, file = 'data/claims-clean-headers.RData')

# word tokenization, tf-idf (slow, lemmatizing)
wide_h <- nlp_fn(claims_clean_headers)
save(wide_h, file = 'data/claims-wide-headers.RData')

# same train/test pages as the baseline
train_df_h <- wide_h %>% filter(.id %in% train_ids)
test_df_h <- wide_h %>% filter(.id %in% test_ids)

# same vocabulary filter rule as the baseline
word_cols_h <- setdiff(names(train_df_h), c('.id', 'bclass'))
page_counts_h <- colSums(train_df_h[, word_cols_h] > 0)
keep_words_h <- word_cols_h[page_counts_h >= min_pages]
length(keep_words_h)

x_train_h <- train_df_h %>% select(all_of(keep_words_h)) %>% as.matrix()
x_test_h <- test_df_h %>% select(all_of(keep_words_h)) %>% as.matrix()
y_train_h <- train_df_h$bclass
y_test_h <- test_df_h$bclass

# PCA on training data only, same k as baseline
pca_h <- prcomp(x_train_h, center = TRUE, scale. = FALSE, rank. = k)
pc_train_h <- pca_h$x
pc_test_h <- predict(pca_h, x_test_h)

# logistic regression on the k components
d_train_h <- data.frame(y = y_train_h, pc_train_h)
fit_h <- glm(y ~ ., data = d_train_h, family = binomial)

# accuracy on held-out test pages
p_h <- predict(fit_h, newdata = data.frame(pc_test_h), type = 'response')
pred_h <- factor(levels(y_train_h)[(p_h > 0.5) + 1], levels = levels(y_train_h))
acc_headers <- mean(pred_h == y_test_h)

# compare
acc_para
acc_headers
nrow(test_df)
nrow(test_df_h)













## FAIR COMPARISON: SAME PAGES IN BOTH VERSIONS
################################################

# fit and score the PCR pipeline for a given tf-idf table and set of pages
run_pcr <- function(w, tr_ids, te_ids, k, min_pages) {
  tr <- w %>% filter(.id %in% tr_ids)
  te <- w %>% filter(.id %in% te_ids)
  cols <- setdiff(names(tr), c('.id', 'bclass'))
  keep <- cols[colSums(tr[, cols] > 0) >= min_pages]
  x_tr <- tr %>% select(all_of(keep)) %>% as.matrix()
  x_te <- te %>% select(all_of(keep)) %>% as.matrix()
  pca <- prcomp(x_tr, center = TRUE, scale. = FALSE, rank. = k)
  fit <- glm(y ~ ., data = data.frame(y = tr$bclass, pca$x), family = binomial)
  p <- predict(fit, newdata = data.frame(predict(pca, x_te)), type = 'response')
  pred <- factor(levels(tr$bclass)[(p > 0.5) + 1], levels = levels(tr$bclass))
  mean(pred == te$bclass)
}

# pages that appear in both versions
common_ids <- intersect(wide$.id, wide_h$.id)
tr_common <- intersect(train_ids, common_ids)
te_common <- intersect(test_ids, common_ids)

acc_para_common <- run_pcr(wide, tr_common, te_common, k, min_pages)
acc_headers_common <- run_pcr(wide_h, tr_common, te_common, k, min_pages)

acc_para_common
acc_headers_common
length(te_common)








## SIGNIFICANCE CHECK
##########################

# standard error (standard deviation of an accuracy estimate)
se_fn <- function(acc, n) sqrt(acc * (1 - acc) / n)

# comparison 1: all available pages
se_para <- se_fn(acc_para, nrow(test_df))
se_head <- se_fn(acc_headers, nrow(test_df_h))
se_diff1 <- sqrt(se_para^2 + se_head^2)
z1 <- (acc_headers - acc_para) / se_diff1

# comparison 2: matched pages
n_c <- length(te_common)
se_para_c <- se_fn(acc_para_common, n_c)
se_head_c <- se_fn(acc_headers_common, n_c)
se_diff2 <- sqrt(se_para_c^2 + se_head_c^2)
z2 <- (acc_headers_common - acc_para_common) / se_diff2

c(se_para = se_para, se_head = se_head, se_diff1 = se_diff1, z1 = z1)
c(se_para_c = se_para_c, se_head_c = se_head_c, se_diff2 = se_diff2, z2 = z2)