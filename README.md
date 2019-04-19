# SentimentAnalysis
Prediction of Sentiments using IMDB Movie Reviews

## Model Overview
- The given dataset is a dataset of 50,000 of IMDB movie reviews. Each movie review has a binary response indicating either a positive (1) or a negative (0) movie review. The end goal of the model is to perform sentiment analysis i.e., to classify a new movie review as either a positive movie review or a negative movie review
- Ridge regression is used to perform the given sentiment analysis task on the given dataset
- The input to the model is a ‘Document-to-term matrix(DTM)’ which is a mathematical matrix that describes the frequency of terms that occur in a collection of documents or in our case, a collection of movie reviews
-	The output of the model will be a probability vector, which will specify the probability of the outcome being a ‘1’ (positive review)

## Vocabulary Customization
•	The create the vocabulary, the reviews are tokenized and converted to lowercase void of any punctuation marks
•	Stopwords are removed, and the vocabulary size is pruned to around 50,000 tokens
•	Finally, a two-sample t-test is used as a simple screening method to obtain the top 3000 words

## Training the Model
•	A N-gram or a bag of words model is used to obtain the final vocabulary. First a normal bag of words model was explored followed by a 4-gram model and then finally the 2-gram model was used. 
•	For preprocessing, the train and test reviews are first converted to lowercase, the punctuations are removed, and are finally tokenized.
•	From the final tokenized words, the respective train and test document-to-term matrices are created
•	The train matrix is then passed into the ridge regression model and prediction probabilities are then obtained using the test matrix
•	The ridge regression used is a cross-validation model where up to 10 folds were used
•	AUC is used a performance metric

## Some Notes
•	A major limitation of ridge regression is that ordinary inference procedures are not applicable, and the exact distributional properties are not known
•	Another limitation is that, there is heavy bias toward zero for large regression coefficients and interpretability, i.e., unimportant coefficients may be shrunken towards zero, but they’re still in the model




