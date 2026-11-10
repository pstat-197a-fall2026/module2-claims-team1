source('scripts/preprocessing.R')
library(tidymodels)

load('data/claims-clean-example.RData')


# ORIGINAL RUN (no headers, using the initial commit data) WITH SLIGHTLY MODIFIED PREPROCESSING 
# (word model baseline, reused from task 1)
#######################

# split by page id so the same pages can be reused for every comparison
set.seed(110122)
partitions <- initial_split(claims_clean, prop = 0.8)
train_ids <- training(partitions)$.id
test_ids <- testing(partitions)$.id

# word tokenization, one row per page, tf-idf columns (loaded from task 1)
load('results/preliminary-task-1/task1-wide-paragraphs.RData')




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


# training accuracy
p_train <- predict(fit, type = 'response')
pred_train <- factor(levels(y_train)[(p_train > 0.5) + 1], levels = levels(y_train))
mean(pred_train == y_train)

# testing accuracy
acc_para <- mean(pred == y_test)
acc_para










## TASK 2: BIGRAMS
####################

# standard error of an accuracy estimate (same as task 1)
se_fn <- function(acc, n) sqrt(acc * (1 - acc) / n)

# --- step 1: log-odds from the word model ---

# training pages get out-of-sample log-odds (5-fold CV on the glm) so the
# second model isn't fed overfit values
set.seed(110122)
folds <- sample(rep(1:5, length.out = nrow(pc_train)))
lo_train <- numeric(nrow(pc_train))
for (f in 1:5) {
  fit_f <- glm(y ~ ., data = d_train[folds != f, ], family = binomial)
  lo_train[folds == f] <- predict(fit_f,
                                  newdata = data.frame(pc_train[folds == f, ]),
                                  type = 'link')
}

# test pages use the full word model fit from the first half
lo_test <- predict(fit, newdata = data.frame(pc_test), type = 'link')

# --- step 2: bigram tokenization ---

min_pages_bi <- 10   # keep bigrams appearing in at least this many training pages

bigrams <- claims_clean %>%
  ungroup() %>%
  select(.id, text_clean) %>%
  filter(.id %in% wide$.id) %>%
  unnest_tokens(bigram, text_clean, token = 'ngrams', n = 2) %>%
  drop_na(bigram) %>%
  separate(bigram, into = c('w1', 'w2'), sep = ' ', remove = FALSE) %>%
  filter(!w1 %in% stop_words$word, !w2 %in% stop_words$word,
         str_length(w1) > 2, str_length(w2) > 2) %>%
  count(.id, bigram, name = 'n')

# vocabulary filter using training pages only
bi_keep <- bigrams %>%
  filter(.id %in% train_ids) %>%
  count(bigram, name = 'df') %>%
  filter(df >= min_pages_bi) %>%
  pull(bigram)
length(bi_keep)  # how many bigrams are left

# tf-idf table, one row per page
bi_wide <- bigrams %>%
  filter(bigram %in% bi_keep) %>%
  bind_tf_idf(term = bigram, document = .id, n = n) %>%
  pivot_wider(id_cols = '.id',
              names_from = 'bigram',
              values_from = 'tf_idf',
              values_fill = 0)

save(bi_wide, file = 'results/preliminary-task-2/task2-wide-bigrams.RData')

# line rows up with the word-model pages (pages with no kept bigrams become 0s)
bi_cols <- setdiff(names(bi_wide), '.id')
align_bi <- function(ids) {
  m <- as.matrix(bi_wide[match(ids, bi_wide$.id), bi_cols])
  m[is.na(m)] <- 0
  m
}
x_bi_train <- align_bi(train_df$.id)
x_bi_test <- align_bi(test_df$.id)

# --- step 3: PCA on bigram data (training pages only) ---

k_bi <- 50   # number of bigram principal components
pca_bi <- prcomp(x_bi_train, center = TRUE, scale. = FALSE, rank. = k_bi)
pc_bi_train <- pca_bi$x
pc_bi_test <- predict(pca_bi, x_bi_test)
colnames(pc_bi_train) <- colnames(pc_bi_test) <- paste0('B', 1:k_bi)

# --- step 4: second logistic regression (log-odds + bigram PCs) ---

d2_train <- data.frame(y = y_train, lo = lo_train, pc_bi_train)
d2_test <- data.frame(lo = lo_test, pc_bi_test)

fit2 <- glm(y ~ ., data = d2_train, family = binomial)

p2 <- predict(fit2, newdata = d2_test, type = 'response')
pred2 <- factor(levels(y_train)[(p2 > 0.5) + 1], levels = levels(y_train))
acc_bigram <- mean(pred2 == y_test)

# --- compare ---

acc_para      # word model alone
acc_bigram    # word log-odds + bigram PCs

# does adding the bigram PCs improve fit on the training pages?
fit_lo <- glm(y ~ lo, data = d2_train, family = binomial)
anova(fit_lo, fit2, test = 'Chisq')

# is the test accuracy difference bigger than noise?
n_t <- length(y_test)
se_diff <- sqrt(se_fn(acc_para, n_t)^2 + se_fn(acc_bigram, n_t)^2)
z_bigram <- (acc_bigram - acc_para) / se_diff
c(se_diff = se_diff, z = z_bigram)

