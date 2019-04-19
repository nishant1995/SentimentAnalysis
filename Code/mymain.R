begin_time = Sys.time()

all = read.table("data.tsv",stringsAsFactors = F,header = T)
splits = read.table("splits.csv", header = T)
s = 3

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
# Cleaning the data
###################################################################################################################


#Removing HTML tags
all$review = gsub('<.*?>', ' ', all$review)

#Removing punctuations and converting to lower case words
all$review = tolower(gsub('[[:punct:]]', '', all$review)) 

###################################################################################################################
# Splitting the data
###################################################################################################################

train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
ytest = test$sentiment 
test$sentiment = NULL

###################################################################################################################
# Further processing
###################################################################################################################

#Creating tokens
train_tokens = itoken(train$review, 
                  tokenizer = word_tokenizer, 
                  ids = train$new_id, 
                  progressbar = FALSE)

test_tokens = itoken(test$review, 
                     tokenizer = word_tokenizer, 
                     ids = test$new_id, 
                     progressbar = FALSE)

# Reading in the vocabulary
vocab = readRDS('MyVocab.rds')

vectorizer = vocab_vectorizer(vocab)
dtm_train  = create_dtm(train_tokens, vectorizer)
dtm_test  = create_dtm(test_tokens, vectorizer)



###################################################################################################################
# Building the model
###################################################################################################################

set.seed(1036)
NFOLDS = 10
mycv = cv.glmnet(x=dtm_train, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)
myfit = glmnet(x=dtm_train, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)
logit_pred = predict(myfit, dtm_test, type = "response")
roc_obj = roc(ytest, as.vector(logit_pred))
auc(roc_obj) 

###################################################################################################################
# Writing the output
###################################################################################################################

output = data.frame(test$new_id, logit_pred)
colnames(output) = c('id','prob')
write.csv(output, paste('Result_', as.character(s), '.txt', sep = ''),row.names = FALSE)

end_time = Sys.time()

Run_Time = end_time - begin_time 










