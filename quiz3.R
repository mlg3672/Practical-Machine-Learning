# Quiz 3 

library(caret)
library(AppliedPredictiveModeling)
library(pgmm)
library(rpart)
library(ElemStatLearn)
# Question 1 ----------
data("segmentationOriginal")
# 1. Subset the data to a training set and testing set based on the Case variable in the data set. 
trainseg<-segmentationOriginal[segmentationOriginal$Case=="Train",]
testseg<-segmentationOriginal[segmentationOriginal$Case=="Test",]
summary(testseg$Case);summary(trainseg$Case)
# 2. Set the seed to 125 and fit a CART model with the rpart method using 
#     all predictor variables and default caret settings. 
set.seed(125)
modFit<-train(Class~., method="rpart", data = trainseg)
print(modFit$finalModel)
library(rattle)
fancyRpartPlot(modFit$finalModel)
# 3. In the final model what would be the final model prediction for 
#    cases with the following variable values:
#   a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 --> PS
#   b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 --> WS
#   c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 --> PS
#   d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 --> Not possible

# Answer: a. PS b. PS c. PS d. PS

# Question 2 ----------
# If K is small in a K-fold cross validation is the bias in the estimate of 
# out-of-sample (test set) accuracy smaller or bigger? If K is small is the 
# variance in the estimate of out-of-sample (test set) accuracy smaller or bigger. 
# Is K large or small in leave one out cross validation?
# Answer: small K, or fewer folds means grab more of data for modeling building
# but make fewer models. So lower variablity but reduced accuracy (larger bias)
# When k is equal to size of dataset it is the same as leave one out.


# Question 3 ----------
# Load the olive oil dataset
# These data contain information on 572 different Italian olive oils 
# from multiple regions in Italy. Fit a classification tree where Area 
# is the outcome variable. Then predict the value of area for the following 
# data frame using the tree command with all defaults
data(olive)
olive<-olive[,-1]
inTrain<-createDataPartition(y= olive$Area, p=0.7, list=FALSE)
trainolive<-olive[inTrain,]
testolive<-olive[-inTrain,]
modFit<-train(Area~., method="rpart", data = trainolive)
modFit
pred<-predict(modFit, testolive)
# diagnostic plot, residuals = diff betweeen real and predicted values)
testolive$predRight<-pred==testolive$Area
qplot(Palmitoleic, Linoleic, colour=predRight, data=testolive, main="new data Predicting")
newdata = as.data.frame(t(colMeans(olive)))
predict(modFit,newdata=newdata)
# Answer: 2.7792 / 2.840708 It is strange because Area should be a qualitative variable
#         - but tree is reporting the average value of Area as a numeric variable in the leaf 
#           predicted for newdata


# Question 4 ----------
# Load the South Africa Heart Disease Data and create training and test sets 
# Then set the seed to 13234 and fit a logistic regression model 
# (method="glm", be sure to specify family="binomial") with Coronary Heart Disease 
# (chd) as the outcome and age at onset, current alcohol consumption, obesity levels, 
# cumulative tabacco, type-A behavior, and low density lipoprotein cholesterol as predictors. 
# Calculate the misclassification rate for your model using this function and a 
# prediction on the "response" scale.
library(ElemStatLearn)
data(SAheart)
set.seed(13234)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
modFit<-train(chd~obesity+age+alcohol+tobacco+typea+ldl+adiposity, method="glm", family="binomial", data = trainSA)
modFit
modFit$finalModel
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
missClass(testSA$chd,predict(modFit, testSA))
missClass(trainSA$chd,predict(modFit, trainSA))
# Answer: Test Set Misclassification: 0.303 Training Set: 0.268

# Question 5 ------
# Load the vowel.train and vowel.test datasets
data(vowel.train)
data(vowel.test)
# Set the variable y to be a factor variable in both the training and test set.
vowel.train$y=as.factor(vowel.train$y)
vowel.test$y=as.factor(vowel.test$y)
# Then set the seed to 33833. 
set.seed(33833)
# Fit a random forest predictor relating the factor variable y to the remaining variables
modFit<-train(y~., method="rf", data = vowel.train)
modFit
pred<-predict(modFit, vowel.test)
vowel.test$predRight<-pred==vowel.test$y
table(pred, vowel.test$y)
qplot(x.1, x.2, colour=predRight, data=vowel.test, main="new data Predicting")
# Calculate the variable importance using the varImp function in the caret package. 
varImp(modFit, decreasing=T)
# What is the order of variable importance?
# Answer: x.2, x.1, x.5, x.6, x.8, x.4, x.9, x.3, x.7,x.10