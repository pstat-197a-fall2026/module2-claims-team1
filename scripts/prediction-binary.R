# *FOR HOW SOMEONE ELSE WOULD USE THE MODEL

## generate binary predictions on the test pages with the saved model
source('scripts/preprocessing-binary.R')
library(e1071)

load('data/claims-test.RData')
binary_model <- readRDS('results/binary-model.rds')

# same preprocessing as training (takes a minute or two)
test_text <- claims_test %>%
  select(.id, text_tmp) %>%
  parse_pages()

x_new <- tfidf_matrix(tokenize_pages(test_text), test_text$.id, binary_model$vocab)

# every test page gets a prediction (pages with no text get the model's default)
bclass_pred <- factor(as.character(predict(binary_model$fit, x_new)),
                      levels = binary_model$levels)

pred_df <- tibble(.id = test_text$.id, bclass.pred = bclass_pred)

load('results/preds-multiclass.RData')


# add mclass.pred from the multiclass model (task b) before saving the final
# results/preds-group1.RData, e.g. pred_df %>% left_join(mclass_df, by = '.id')

pred_df <- pred_df %>% left_join(mclass_df, by = '.id')

save(pred_df, file = 'results/preds.RData')

#table(pred_df$bclass.pred)
#table(pred_df$mclass.pred)
