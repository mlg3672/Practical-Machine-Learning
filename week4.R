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
library(quantmod)
from.dat <- as.Date("01/01/08",format = "%m/%d/%y")
to.dat<- as.Date("12/31/13",format = "%m/%d/%y")
getSymbols("GOOG",src = "google", from = from.dat, to= to.dat)
head(GOOG)
index(GOOG)<-as.POSIXct(index(GOOG))
mGOOg <- to.monthly(GOOG[,-5]) 
googOpen <-Op(mGOOg)
ts1 <- ts(googOpen, frequency = 12)
plot(ts1,xlab="Years+1",ylab = "GOOG")
plot(decompose(ts1),xlab="Years+1")

# moving average
library(forecast)
ts1Train<-window(ts1,start=1,end=5)
ts1Test<-window(ts1,start=5,end=(7-0.01))
plot(ts1Train)
lines(ma(ts1Train,order=3),col="red")

# exponential smoothing
ets1 <- ets(ts1Train,model="MMM")
fcast <-forecast(ets1)
plot(fcast,col="black")
lines(ts1Test,col="red")
# accuracy of forecast RMSE etc.
accuracy(fcast,ts1Test)

# Notes from book ---
# 
# Moving Averages of Moving Averages 
#   m = 4 quarterly data
#   m = 12 monthly data
#   m = 7 weekly data
# see x12 package for X-12-ARIMA decomposition

# Lecture 27 - Unsupervised Prediction -----
# Key Ideas : when don't know the labels of predictors...
#             (1) build predictor by creating, naming clusters
#             (2) in a new dataset predict clusters
# exploratory technique, beware of overinterpretation
library(caret)
library(ggplot2)
data(iris)
inTrain<-createDataPartition(y= iris$Species, p=0.7, list=FALSE)
training<-iris[inTrain,]
testing<-iris[-inTrain,]
dim(training); dim(testing)
kMeans1<- kmeans(subset(training, select=-c(Species)),centers=3)
training$clusters<- as.factor(kMeans1$cluster)
qplot(Petal.Width, Petal.Length, colour=clusters, data=training)
table(kMeans1$cluster,training$Species)
modFit<-train(clusters~.,method="rpart", data=subset(training, select=-c(Species)))
table(predict(modFit,training),training$Species)
testClusterPred<-predict(modFit,testing)
table(testClusterPred,testing$Species)
