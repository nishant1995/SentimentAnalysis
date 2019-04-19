###################################################################################################################
# Function to check if packages are installed 
###################################################################################################################


check.packages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}


packages = c('tm','text2vec', 'data.table', 'magrittr','iterators','glmnet', 'pROC', 'slam')
check.packages(packages)

###################################################################################################################
# Creating the vocabulary
###################################################################################################################

all = read.table("data.tsv",stringsAsFactors = F,header = T)

all$review = gsub('<.*?>', ' ', all$review)
all$review = tolower(gsub('[[:punct:]]', '', all$review))

tokens = itoken(all$review, 
                tokenizer = word_tokenizer, 
                ids = all$new_id, 
                progressbar = FALSE)

stop_words = stopwords(kind = 'en')

vocab = create_vocabulary(tokens, stopwords = stop_words, ngram = c(1L,2L))

prune_vocab = prune_vocabulary(vocab, 
                               term_count_min = 5, 
                               doc_proportion_max = 0.5, 
                               vocab_term_max = 50000)

vectorizer = vocab_vectorizer(prune_vocab)
dtm_all  = create_dtm(tokens, vectorizer)

v.size = dim(dtm_all)[2]
senti = all$sentiment

dtm_all = as.simple_triplet_matrix(dtm_all)
summ = simple_triplet_zero_matrix(nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(dtm_all[senti==1, ], mean)
summ[,2] = colapply_simple_triplet_matrix(dtm_all[senti==1, ], var)
summ[,3] = colapply_simple_triplet_matrix(dtm_all[senti==0, ], mean)
summ[,4] = colapply_simple_triplet_matrix(dtm_all[senti==0, ], var)
n1=sum(senti); 
n=length(senti)
n0= n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)

myp = as.matrix(myp)

id = order(abs(myp), decreasing=TRUE)[1:3000]


rownames(prune_vocab) = NULL
vocab = prune_vocab[id,]
saveRDS(vocab, 'MyVocab.rds')
