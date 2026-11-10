# preprocessing functions for the binary classifier (primary task a)
# intended to be sourced by binary-model-training.R and prediction-binary.R
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(Matrix)

## clean_text()
# same cleaning steps as parse_fn() in preprocessing.R, applied to any text
# removes URLs, emails, punctuation, and digits, AND turns everything to lowercase
clean_text <- function(x){
  x <- x[!is.na(x)]
  x %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', '[[:punct:]]', 'nbsp',
                            '[[:digit:]]', '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

## parse_page()
# pull all readable text from a page: title, meta description, headers,
# paragraphs, and list/table/quote text (paragraphs alone miss ~10% of pages)
parse_page <- function(.html){
  if (is.na(.html) || !str_detect(.html, '<!')) return('')
  tryCatch({
    pg <- read_html(.html)
    meta <- pg %>%
      html_elements('meta[name="description"], meta[property="og:description"]') %>%
      html_attr('content')
    body <- pg %>%
      html_elements('title, h1, h2, h3, h4, h5, h6, p, li, td, blockquote') %>%
      html_text2()
    clean_text(c(meta, body))
  }, error = function(e) '')
}

# apply to a data frame with columns .id and text_tmp
parse_pages <- function(.df){
  .df %>%
    mutate(text_clean = map_chr(text_tmp, parse_page)) %>%
    select(-text_tmp)
}

# word tokens: stop words removed, lemmatized, at least 3 characters
stop_vec <- str_remove_all(stop_words$word, '[[:punct:]]')

## tokenize_pages()
# splits text into words, drops stop words, lemmatizes (ie "injuries" becomes "injury"), counts words per page
tokenize_pages <- function(.df){
  .df %>%
    select(.id, text_clean) %>%
    unnest_tokens(token, text_clean, token = 'words') %>%
    filter(!token %in% stop_vec) %>%
    mutate(token = lemmatize_words(token)) %>%
    filter(str_length(token) > 2) %>%
    count(.id, token, name = 'n')
}

## fit_vocab()
# picks words that appear on >=5 training pages and computes their idf weights
# vocabulary and idf weights, learned from training pages only
fit_vocab <- function(tokens, min_docs = 5){
  n_docs <- n_distinct(tokens$.id)
  tokens %>%
    count(token, name = 'df') %>%
    filter(df >= min_docs) %>%
    mutate(idf = log((1 + n_docs) / (1 + df)) + 1)
}


## tfidf_matrix()
# turns the word counts into the tf-idf matrix, one row per page
# sparse tf-idf matrix (log term frequency, rows scaled to unit length);
# one row per id, in order, including pages with no usable text
tfidf_matrix <- function(tokens, ids, vocab){
  t <- tokens %>%
    filter(.id %in% ids) %>%
    inner_join(vocab, by = 'token') %>%
    mutate(i = match(.id, ids),
           j = match(token, vocab$token),
           x = (1 + log(n)) * idf)
  m <- sparseMatrix(i = t$i, j = t$j, x = t$x,
                    dims = c(length(ids), nrow(vocab)),
                    dimnames = list(ids, vocab$token))
  norms <- sqrt(rowSums(m^2))
  norms[norms == 0] <- 1
  as(Diagonal(x = 1 / norms) %*% m, 'CsparseMatrix')
}
