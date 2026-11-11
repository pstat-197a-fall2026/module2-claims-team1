# *FOR HOW SOMEONE ELSE WOULD USE THE MULTICLASS MODEL

## generate multiclass predictions on the test pages with the saved model
source('scripts/preprocessing-binary.R')
library(e1071)
library(tidyverse)

load('data/claims-test.RData')
multiclass_model <- readRDS('results/multiclass-model.rds')

# same preprocessing as training (takes a minute or two)
test_text <- claims_test %>%
  select(.id, text_tmp) %>%
  parse_pages()

x_multi <- tfidf_matrix(tokenize_pages(test_text), test_text$.id, multiclass_model$vocab)

# every test page gets a prediction (pages with no text get the model's default)
mclass_pred <- factor(as.character(predict(multiclass_model$fit, x_multi)),
                      levels = multiclass_model$levels)

mclass_df <- tibble(.id = test_text$.id, mclass.pred = mclass_pred)

save(mclass_df, file = 'results/preds-multiclass.RData')

#table(mclass_df$mclass.pred)