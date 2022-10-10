library(sentimentr)
library(ggplot2)
wine <- read.csv('/Users/xiaojieqian/758T Data mining/HW/Wine Data for Lab.csv')
#
wine$description <- as.character(wine$description)
text = get_sentences(wine$description) # sentence boundary disambiguation 
sentiment = sentiment_by(text) 
sentiment = data.frame(sentiment)
wine <- cbind(wine, sentiment)
#Q1
qplot(wine$ave_sentiment,geom="histogram",binwidth=0.1,main="Wines Sentiment scores Histogram")
summary(wine$ave_sentiment)
by(wine$ave_sentiment,wine$country=='New Zealand',summary)
by(wine$ave_sentiment,wine$price==c(20,29.99),summary)
#Q2
fit <- lm(points~ave_sentiment, data=wine)
summary(fit)
fit2 <- lm(points~ave_sentiment+price, data=wine)
summary(fit2)
#Q3
library(tidyverse)
str(wine)
wine <- arrange(wine,desc( ave_sentiment))
wine <- wine %>%
  mutate(lable=ifelse(wine$ave_sentiment>wine$ave_sentiment[0.4*nrow(wine)],
                      "positive","negative"))

## --------------------------Pre-processsing of text-----------------------

library(e1071)
library(SparseM)
library(tm)
# library(SnowballC);
usableText <- iconv(wine$description, "ASCII", "UTF-8", sub="")
winevector <- as.vector(usableText);    # Create vector
winesource <- VectorSource(winevector); # Create source
winecorpus <- Corpus(winesource);       # Create corpus
#
# PERFORMING THE VARIOUS TRANSFORMATIONS on "traincorpus" and "testcorpus" DATASETS 
# SUCH AS TRIM WHITESPACE, REMOVE PUNCTUATION, REMOVE STOPWORDS.
winecorpus <- tm_map(winecorpus,content_transformer(stripWhitespace));
winecorpus <- tm_map(winecorpus,content_transformer(tolower));
winecorpus <- tm_map(winecorpus, content_transformer(removeWords),stopwords("english"));
winecorpus <- tm_map(winecorpus,content_transformer(removePunctuation));
winecorpus <- tm_map(winecorpus,content_transformer(removeNumbers));
# 
# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
winecorpus <- tm_map(winecorpus, content_transformer(removeNumPunct))
# 
# remove URLs
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
winecorpus <- tm_map(winecorpus, content_transformer(removeURL))
#
#
# Create TermDocumentMatrix
tdm1 <- TermDocumentMatrix(winecorpus)
tdm1 = removeSparseTerms(tdm1, 0.95)
#
inspect(tdm1)
# CREATE TERM DOCUMENT MATRIX
dfmatrix <- t(tdm1);

# TRAIN NAIVE BAYES MODEL
df2 <- data.frame(as.matrix(dfmatrix))
#
set.seed(12345)
inTrain <- sample(nrow(df2),0.7*nrow(df2))
train_input<-df2[inTrain,]
test_input<-df2[-inTrain,]
train_output<-wine[inTrain,]$lable
test_output<-wine[-inTrain,]$lable
##
library(e1071)
NB <- naiveBayes(train_input,train_output)
# PREDICTION
Predictions <- predict(NB,test_input)
(confusion = table(test_output,Predictions))
accuracy <- (confusion[1,1]+confusion[2,2])/sum(confusion)
accuracy
##
set.seed(12345)
train_output_2<-wine[inTrain,]$ave_sentiment
test_output_2<-wine[-inTrain,]$ave_sentiment
model_lm<-lm(train_output_2~.,data=train_input)
pred_lm<-predict(model_lm,newdata=test_input)
(RMSE_lm<-sqrt(mean((test_output_2 - pred_lm)^2)))
#Q4
# inspect frequent words
(freq.terms <- findFreqTerms(tdm1, lowfreq = 15))
#
term.freq <- rowSums(as.matrix(tdm1))
term.freq <- subset(term.freq, term.freq >= 15)
df3 <- data.frame(term = names(term.freq), freq = term.freq)
library(ggplot2)
ggplot(df3, aes(x = term, y = freq)) + geom_bar(stat = "identity") +
  xlab("Terms") + ylab("Count") + coord_flip()

## hclust ####
matrix1 <- as.matrix(tdm1)
distMatrix <- dist(scale(matrix1))
fit3 <- hclust(distMatrix, method="ward.D2")
# plot dendrogram ####
plot(fit3, cex=0.9, hang=-1,
     main="Word Cluster Dendrogram")
# cut tree
#rect.hclust(fit, k=5)
#(groups <- cutree(fit, k=5))
## Word Cloud
library(wordcloud)
library(RColorBrewer)
m <- as.matrix(tdm1)
# calculate the frequency of words and sort it by frequency
word.freq <- sort(rowSums(m), decreasing = T)
# colors
pal <- brewer.pal(9, "BuGn")
pal <- pal[-(1:4)]
#
# plot word cloud
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 1,
          random.order = F, colors = pal)
#
