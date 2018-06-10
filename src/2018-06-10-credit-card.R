setwd('/Users/johnsonwu/git/deepai')
library(pROC)
library(microbenchmark)

set.seed(42)

credit.card.data <- read.csv('data/creditcard.csv')