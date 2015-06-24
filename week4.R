#Week 4

# Lecture 24 - Regularized Regression -----
library(ElemStatLearn)
data("prostate")
str(prostate)
small = prostate[1:5,]
lm(lpsa~.,data=small)
# Lecture 25 - Combining Predictors -----
library(ISLR)
data(Wage)
library(ggplot2); library(caret)
Wage<-subset(Wage, select =-c(logwage))
# create a Building dataset and validation data set
inBuild<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
validation<-Wage[inBuild,]
buildData<-Wage[-inBuild,]
inTrain<-createDataPartition(y=buildData$wage, p=0.7, list =FALSE)
training<-buildData[inTrain,]
testing<-buildData[-inTrain,]
dim(training);dim(testing);dim(validation)
mod1<-train(wage~.,method="glm", data=training)
mod2<-train(wage~.,method="rf", data=training,
            trControl = trainControl(method = "cv"), number = 3)
pred1<-predict(mod1,testing)
pred2<-predict(mod2,testing)
qplot(pred1, pred2,colour=wage,data=testing)
# fit a model that combines predictors using test set
predDF<-data.frame(pred1,pred2,wage=testing$wage)
combModFit<-train(wage~., method="gam",data=predDF)
# calculate RMSE
combPred<-predict(combModFit,predDF)
sqrt(sum((pred1-testing$wage)^2))
sqrt(sum((pred2-testing$wage)^2))
sqrt(sum((combPred-testing$wage)^2))
# test with validation data set
pred1V<-predict(mod1,validation)
pred2V<-predict(mod2,validation)
predVDF<-data.frame(pred1V,pred2V)
combPredV<-predict(combModFit,predVDF) # doesnt work mismatch rows
# calculate RMSE
sqrt(sum((pred1V-validation$wage)^2))
sqrt(sum((pred2V-validation$wage)^2))
sqrt(sum((combPredV-validation$wage)^2)) # doesn't work!

# Lecture 26 - Forecasting ----------
# Lecture 27 - Unsupervised Prediction -----
