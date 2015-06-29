# Quiz #4
library(AppliedPredictiveModeling)
library(caret)
library(lubridate)
library(pgmm)
library(ElemStatLearn)
library(rpart)
library(gbm)
library(forecast)
library(e1071)

# Question 1 ----
# Load the vowel.train and vowel.test datasets
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 
# Set the variable y to be a factor variable in both the 
# training and test set. Then set the seed to 33833. 
vowel.train$y=as.factor(vowel.train$y)
vowel.test$y=as.factor(vowel.test$y)
set.seed(33833)
# Fit (1) a random forest predictor relating the factor 
# variable y to the remaining variables and (2) a boosted 
# predictor using the "gbm" method. Fit these both with the 
# train() command in the caret package. 
mod1<-train(y~., method="rf", data = vowel.train)
mod2<-train(y~., method="gbm", data = vowel.train,verbose = FALSE)
pred1<-predict(mod1,vowel.test)
pred2<-predict(mod2,vowel.test)
# What are the accuracies for the two approaches on the test data set? 
# What is the accuracy among the test set samples where the two methods agree?
predSame<-pred1==pred2
accurPred = function(prediction,values){sum((prediction == values))/length(values)}
accurPred(pred1[predSame], vowel.test$y[predSame])
accurPred(pred1, vowel.test$y)
accurPred(pred2, vowel.test$y)
# Answer: rf : 0.606, gbm : 0.525, both : 0.627


# Question 2 ----
# Load the Alzheimer's data using the following commands
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
class(adData$diagnosis)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
# Set the seed to 62433 and predict diagnosis with all the other variables 
# using a random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model. 

set.seed(62433)
modrf<-train(diagnosis~., method="rf", data = training)
modgbm<-train(diagnosis~., method="gbm", data = training, verbose = FALSE)
modlda<-train(diagnosis~., method="lda", data = training)
predrf<-predict(modrf,testing)
predgbm<-predict(modgbm,testing)
predlda<-predict(modlda,testing)
# Stack the predictions together using random forests ("rf"). 
predDF<-data.frame(predrf,predgbm,predlda,diagnosis=testing$diagnosis)
combModFit<-train(diagnosis~., method="rf",data=predDF)
# What is the resulting accuracy on the test set? 
combPred<-predict(combModFit,predDF)
accurPred(combPred, testing$diagnosis)
# Is it better or worse than each of the individual predictions?
accurPred(predrf, testing$diagnosis)
accurPred(predgbm, testing$diagnosis)
accurPred(predlda, testing$diagnosis)
# Answer: rf 0.780, gbm 0.805, lda 0.768, comb 0.817

# Question 3 ----
# Load the concrete data with the commands:
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
# Set the seed to 233 and fit a lasso model to predict Compressive Strength. 
set.seed(233)
library(elasticnet)
modFit<-train(CompressiveStrength~., 
              method="lasso",
              data=training, 
              metric="RMSE",
              tuneLength = 10, 
              trControl = trainControl(method="cv",number=10))
pred<-predict(modFit,testing)
# Which variable is the last coefficient to be set to zero as the penalty increases? 
# (Hint: it may be useful to look up ?plot.enet).
plot.enet(modFit$finalModel,xvar="penalty",use.color = T)
# Answer: CoarseAggregate is the first, Cement is the last

# Question 4 ----
# Load the data on the number of visitors to the instructors blog 
library(lubridate)  # For year() function below
dat = read.csv("gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
plot(tstrain,xlab="Date",ylab = "Tumblrvisits")
# Fit a model using the bats() function in the forecast package to the training time series. 
modBat<-bats(tstrain)
fcast <-forecast(modBat,h=length(testing$visitsTumblr),level=c(80,95))
plot(fcast,col="black")
# Then forecast this model for the remaining time points. 
ts1Test<-ts(testing$visitsTumblr,start=366)
lines(ts1Test,col="red")
# For how many of the testing points is the true value within the 95% prediction interval bounds?
accurTime= function(forecast,value) { sum((forecast$upper[,2]>value)*(forecast$lower[,2]<value))/length(value)}
accurTime(fcast,testing$visitsTumblr)
# Answer = 0.96

# Question 5 ----
#Load the concrete data with the commands:
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
# Set the seed to 325 and fit a support vector machine using the 
# e1071 package to predict Compressive Strength using the default settings. 
set.seed(325)
modsvm = svm(CompressiveStrength~., data=training)
# OR
fitsvm <- train(CompressiveStrength ~., data=training, method="svmRadial")
predsvm = predict(modsvm,testing)

# Predict on the testing set. What is the RMSE?
accuracy(predsvm, testing$CompressiveStrength)[2]
# OR
RMSE<-sqrt(sum((predsvm-testing$CompressiveStrength)^2)/length(predsvm))
RMSE
# Answer = 6.715