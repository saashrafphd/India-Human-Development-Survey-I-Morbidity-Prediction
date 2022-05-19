install.packages("caret")
install.packages("Matrix")
install.packages("xgboost")
install.packages("ROCR")
mydata <- read.csv(file.choose(), header = T)

# Select and rearrange the order of the features we'll be using
#mydata <- mydata[, c('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin')]

# Convert the Pclass feature to an ordered factor 1 < 2 < 3
#mydata$Pclass <- factor(mydata$Pclass, ordered=TRUE)

str(mydata) # View the structure of our dataset
head(mydata) # View the first six rows of the dataset
sapply(mydata, function(x) sum(is.na(x)))
library(caret)

# Set the seed to create reproducible train and test sets
set.seed(300)

# Create a stratified random sample to create train and test sets
# Reference the outcome variable
trainIndex   <- createDataPartition(mydata$Major_Morbidity, p=0.70, list=FALSE, times=1)
train        <- mydata[ trainIndex, ]
test         <- mydata[-trainIndex, ]
# Create separate vectors of our outcome variable for both our train and test sets
# We'll use these to train and test our model later
train.label  <- train$Major_Morbidity
test.label   <- test$Major_Morbidity
library(Matrix)

# Create sparse matrixes and perform One-Hot Encoding to create dummy variables
dtrain  <- sparse.model.matrix(Major_Morbidity ~ .-1, data=train)
dim(dtrain)
dtest   <- sparse.model.matrix(Major_Morbidity ~ .-1, data=test)
# View the number of rows and features of each set
dim(dtrain)
dim(dtest)
library(xgboost)

# Set our hyperparameters
param <- list(objective   = "binary:logistic",
              eval_metric = "error",
              max_depth   = 7,
              eta         = 0.1,
              gammma      = 1,
              colsample_bytree = 0.5,
              min_child_weight = 1)

set.seed(1234)

# Pass in our hyperparameteres and train the model 
system.time(xgb <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = 500,
                           print_every_n = 100,
                           verbose = 1))
# Create our prediction probabilities
pred <- predict(xgb, dtest)

# Set our cutoff threshold
pred.resp <- ifelse(pred >= 0.7, 1, 0)
str(pred.resp)

# Create the confusion matrix
confusionMatrix(pred.resp, test$Major_Morbidity )
# Get the trained model
model <- xgb.dump(xgb, with_stats=TRUE)

# Get the feature real names
names <- dimnames(dtrain)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model=xgb)[0:12] # View top 12 most important features

# Plot
xgb.plot.importance(importance_matrix)

library(ROCR)

# Use ROCR package to plot ROC Curve
xgb.pred <- prediction(pred, test.label)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")

plot(xgb.perf,
     colorize=TRUE,
     lwd=1,
     main="ROC Curve w/ Thresholds",
     text.adj=c(-0.5, 0.5),
     text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.1))
axis(2, at=seq(0, 1, by=0.1))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")
auc_ROCR <- performance(xgb.pred, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
auc_ROCR
