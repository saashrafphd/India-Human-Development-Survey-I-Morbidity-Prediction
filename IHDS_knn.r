install.packages("ISLR")
install.packages("caret")
install.packages("pROC")
install.packages('e1071', dependencies=TRUE)
set.seed(300)
mydata <- read.csv(file.choose(), header = T)
#Spliting data as training and test set. Using createDataPartition() function from caret
indxTrain <- createDataPartition(mydata$Major_Morbidity,p = 0.70,list = FALSE)
training <- mydata[indxTrain,]
testing <- mydata[-indxTrain,]

#Checking distibution in origanl data and partitioned data
prop.table(table(training$Major_Morbidity)) * 100
prop.table(table(testing$Major_Morbidity)) * 100
prop.table(table(mydata$Major_Morbidity)) * 100


set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Major_Morbidity ~ ., data = training, method = "knn", trControl = ctrl, tuneLength = 20)
knnFit
plot(knnFit)
knnPredict <- predict(knnFit,newdata = testing )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, testing$Major_Morbidity )
mean(knnPredict == testing$Major_Morbidity)
#Now verifying 2 class summary function

ctrl <- trainControl(method="repeatedcv",repeats = 3,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Major_Morbidity ~ ., data = training, method = "knn", trControl = ctrl,, tuneLength = 20)
knnFit
plot(knnFit, print.thres = 0.5, type="S")
knnPredict <- predict(knnFit,newdata = testing )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, testing$Major_Morbidity )
mean(knnPredict == testing$Major_Morbidity)
library(pROC)
knnPredict <- predict(knnFit,newdata = testing , type="prob")
knnROC <- roc(testing$Major_Morbidity,knnPredict[,"Yes"])
knnROC
plot(knnROC, type="S")

