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
# Fit (1) a random forest predictor relating the factor 
# variable y to the remaining variables and (2) a boosted 
# predictor using the "gbm" method. Fit these both with the 
# train() command in the caret package. 

# What are the accuracies for the two approaches on the test data set? 
# What is the accuracy among the test set samples where the two methods agree?

# Question 2 ----
# Load the Alzheimer's data using the following commands
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
# Set the seed to 62433 and predict diagnosis with all the other variables 
# using a random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model. 
# Stack the predictions together using random forests ("rf"). 
# What is the resulting accuracy on the test set? Is it better or worse than 
# each of the individual predictions?

# Question 3 ----
# Load the concrete data with the commands:
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
# Set the seed to 233 and fit a lasso model to predict Compressive Strength. 
# Which variable is the last coefficient to be set to zero as the penalty increases? 
# (Hint: it may be useful to look up ?plot.enet).

# Question 4 ----
# Load the data on the number of visitors to the instructors blog 
library(lubridate)  # For year() function below
dat = read.csv("~/Desktop/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
# Fit a model using the bats() function in the forecast package to the training time series. 
# Then forecast this model for the remaining time points. 
# For how many of the testing points is the true value within the 95% prediction interval bounds?

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
# Predict on the testing set. What is the RMSE?