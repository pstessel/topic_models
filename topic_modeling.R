# A topic model for movie reviews
# http://cpsievert.github.io/LDAvis/reviews/reviews.html

#devtools::install_github("cpsievert/LDAvisData")

# Create Reviews
if (!file.exists("data-raw/reviews")) {
  tmp <- tempfile(fileext = ".tar.gz")
  download.file("http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz", 
                tmp, quiet = TRUE)
  untar(tmp, exdir = "data-raw/reviews")
  unlink(tmp)
}

path <- file.path("data-raw", "reviews", "txt_sentoken")
pos <- list.files(file.path(path, "pos"))
neg <- list.files(file.path(path, "neg"))
pos.files <- file.path(path, "pos", pos)
neg.files <- file.path(path, "neg", neg)
all.files <- c(pos.files, neg.files)
txt <- lapply(all.files, readLines)
nms <- gsub("data-raw/reviews/txt_sentoken", "", all.files)
reviews <- setNames(txt, nms)
reviews <- sapply(reviews, function(x) paste(x, collapse = " "))

save(reviews, file = "reviews.rdata", compress = "xz")

data(reviews, package = "LDAvisData")

#####################################################################

# read in some stopwords:
library(tm)
stop_words <- stopwords("SMART")

# pre-processing:
reviews <- gsub("'", "", reviews)  # remove apostrophes
reviews <- gsub("[[:punct:]]", " ", reviews)  # replace punctuation with space
reviews <- gsub("[[:cntrl:]]", " ", reviews)  # replace control characters with space
reviews <- gsub("^[[:space:]]+", "", reviews) # remove whitespace at beginning of documents
reviews <- gsub("[[:space:]]+$", "", reviews) # remove whitespace at end of documents
reviews <- tolower(reviews)  # force to lowercase

# tokenize on space and output as a list:
doc.list <- strsplit(reviews, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)
term.table

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
vocab <- names(term.table)
vocab

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents) # number of documents (2,000)
D
W <- length(vocab) # number of terms in the vocab (14,567)
W
doc.length <- sapply(documents, function(x) sum(x[2, ])) # number of tokens per document
doc.length
N <- sum(doc.length) # total number of tokens in the data
N
term.frequency <- as.integer(term.table) # frequencies of terms in the corpus

# MCMC and model tuning parameters
K <- 20
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab,
                                   num.iterations = G, alpha = alpha,
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = T)
t2 <- Sys.time()
t2 - t1
fit

