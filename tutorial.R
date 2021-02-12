## Tutorial: https://ucdavisdatalab.github.io/workshop-nlp-healthcare/exploratory-data-analysis.html
# repo:
# instructors: UC Davis; Arthur Koehl and Wesley Brooks


library(tm)
library(ggplot2)
library(Matrix)

# Link to the data
url_data <- "https://ucdavisdatalab.github.io/workshop-nlp-healthcare/abstracts.csv"

# read the data in
data <- read.csv(url_data, stringsAsFactors = FALSE, encoding = "utf-8")

# Preprocessing ----
# Create corpus
corpus <- Corpus(VectorSource(data$text))
inspect(head(corpus))

# remove case
corpus <- tm_map(corpus, tolower)

# remove punctuation
corpus <- tm_map(corpus, removePunctuation, ucp=TRUE) # ucp remove additional characters (remove strange characters)

# Remove numbers
corpus <- tm_map(corpus, removeNumbers)

# Remove stop words
corpus <- tm_map(corpus, removeWords, stopwords("english"))
## Note you could remove anywords you pass as a vector, e.g: corpus <- tm_map(corpus, removeWords, c("and", "or", "red"))

corpus[[1]]$content

# Bags of words ----

# create it
dtm <- DocumentTermMatrix(corpus)
inspect(dtm)

# make it a standrad matric (rather than a soarse one)
dtm_mat <- as.matrix(dtm)

# document lengths; word counts;
document_lenghts <- rowSums(dtm_mat)
head(sort(document_lenghts, decreasing = TRUE), n=10)

# Term Frequency Inverse Document Frequency ----
# TF-IDF formula: tfidft(t,d,D) = tf(t,d) * idf(t,D) where D is the Document frequency; thus weight inversly

# weighted dtm
tfidf_dtm <- weightTfIdf(dtm, normalize=TRUE)
inspect(tfidf_dtm)

#compare most important terms

tf <- dtm_mat[10, ]
most_importnatn_tf <- head(sort(tf, decreasing=TRUE), n=10)
most_importnatn_tf

# make a simple matru=ix (since dataset is small)
tfmat <- as.matrix(tfidf_dtm)


tfidf <- tfidf_dtm_mat[10,]
most_importnant_tfidf <- head(sort(tfidf, decreasing = TRUE), n=10)
most_importnant_tfidf


# Exploratory Data analaysis -----

## plot tfidf ----
hist(tfmat[,1], main=paste0("uses of '", colnames(tfmat)[[1]], "'"))
hist(tfmat[,2], main=paste0("uses of '", colnames(tfmat)[[2]], "'"))

# Plot the first two most frequent terms
plot(tfmat[,1:2], main="joint uses of adverse and absolute")

## pca-calc -----

# rotate the TF-IDF so the columns are articles
articles <- t(tfmat)

# calculate PCA on the rotated TF-IDF
pca <- prcomp(articles, center=TRUE, scale=TRUE)

# calculate PCA on the rotated TF-IDF
pca_mat <- as.data.frame(pca$rotation)

# plot the 2 first components
with(pca_mat, plot(PC1, PC2))

# identify the order of document align 1st component
# create the index
indx <- order(pca_mat$PC1)

# Select the first titles
data$title[head(indx)]
# last
data$title[tail(indx)]

# identify the order of document align 1st component
# create the index
indx2 <- order(pca_mat$PC2)

# Select the first titles
data$title[head(indx2)]
# last
data$title[tail(indx2)]

# identify the order of document align 1st component
# create the index
indx3 <- order(pca_mat$PC3)

# Select the first titles
data$title[head(indx3)]
# last
data$title[tail(indx3)]

## Identify the entire
# Source function from Github
source(url("https://ucdavisdatalab.github.io/workshop-nlp-healthcare/top_terms.R"))

# cbind data and the PCA rotation
plotdata <- cbind(data, pca_mat)

# identify top terms for each document and attach them to the plotdata
plotdata[[ 'top_terms' ]] <- top_terms( tfidf_dtm )

# plot
ggplot(plotdata) + aes(x=PC2, y=PC3, label=top_terms) + geom_point()

# replace points by terms
ggplot(plotdata) + aes(x=PC2, y=PC3, label=top_terms) + geom_text(check_overlap=TRUE)

# plot explained cariance by the axes

# Percent of Td-idf matrix that is explained
head(pca$sdev)


plot(100 * cumsum(pca$sdev^2) / sum(pca$sdev^2), type='l', bty='n',
     ylab="% total variance explained", xlab="Number of components")

