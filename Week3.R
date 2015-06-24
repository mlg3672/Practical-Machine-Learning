# Week 3 
# Lecture 19- Predicting with Trees -----
# key ideas: (1) iteratively split var into groups
#             (2) evaluate homogenity in groups
#              repeat (1) until groups are too small or homogenous
# Pros : easy to interpret, better performance
# Cons: w/ xvalidation leads to overfitting
# Measures of impurity ::
# 1. Missclassificaiton Error (0=no error, 0.5=no homogenity)
# 2. Gini Index (0=purity, 0.5=no purity)
# 3. Deviance of Information Gain (DoIG) (1=no purity, 0=purity)

# Example: 1 of 16 samples misclassified
# Misclassification Error: 1/16 = 0.06
# Gini : 1 - [(1/16)^2 + (15/16)^2] = 0.12
# DoIG : - [(1/16)log2(1/16) + (15/16)log2(15/16)]  = 0.34

data("iris")
library(ggplot2)
names(iris)
table(iris$Species)
inTrain<-createDataPartition(y= iris$Species, p=0.7, list=FALSE)
training<-iris[inTrain,]
testing<-iris[-inTrain,]
dim(training);dim(testing)
qplot(Petal.Width, Sepal.Width, colour = Species, data = training)
library(caret)
library(rpart)
library(rpart.plot)
modFit<-train(Species~., method="rpart", data = training)
print(modFit$finalModel)
plot(modFit$finalModel, uniform=TRUE, main= "Classification Tree")
text(modFit$finalModel, use.n= TRUE, all = TRUE, cex = 0.8)
library(rattle)
fancyRpartPlot(modFit$finalModel)
predict(modFit, newdata = testing)

# other tree building options - "party", "tree"

# Lecture 20- Bagging -------------------
# basic ideas: (1) average models for better predictor
#              (2) resample cases and recalculate predictions
#              (3) average or majority vote from predictors
library(ElemStatLearn)
data(ozone, package = "ElemStatLearn")
ozone<- ozone[order(ozone$ozone),]
head(ozone)
ll<-matrix(NA, nrow = 10, ncol = 155)
for (i in 1:10) {
  ss<-sample( 1: dim(ozone)[1],replace = T)
  ozone0<-ozone[ss,]
  ozone0<-ozone0[order(ozone0$ozone),]
  loess0<-loess(temperature~ozone, data = ozone0, span = 0.2)
  ll[i,]<-predic(loess0,newdata= data.frame(ozone=1:155))
}

plot(ozone$ozone,ozone$temperature, pch=19,cex = 0.5)
for (i in 1:10) { lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155, apply(ll,2,mean),col="red",lwd=2)

predictors = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag<- bag(predictors, temperature, B=10, bagControl = 
                bagControl(fit = ctreeBag$fit(),
                           predict = ctreeBag$pred(),
                           aggregate = ctreeBag$aggregate()))
plot(ozone$ozone, temperature, col = "lightgrey", pch=19)
points(ozone$ozone, predict(treebag$fit[[1]]$fit,predictors),pch=19, col = "red")
points(ozone$ozone, predict(treebag,predictors),pch = 19, col = "blue")
# notes:  bagging is useful for nonlinear models
#         other bagging models: bagEarth (earth), treebag-(ipred, plyr), bagFDA-(packages: earth,mda)

# Lecture 21- Random Forests ------------
# Basic ideas : (1) bootstrap samples, (2) at each split bootstrap variable
#               (3) grow multiple trees (4) vote 
data("iris")
library(ggplot2)
inTrain<-createDataPartition(y= iris$Species, p=0.7, list=FALSE)
training<-iris[inTrain,]
testing<-iris[-inTrain,]
library(caret)
modFit<-train(Species~., method="rf", data = training, prox = T)
modFit
getTree(modFit, k=2) # see second tree
irisP<- classCenter(training[,c(3,4)],training$Species, modFit$finalModel$prox)
irisP$Species<-rownames(irisP)
p<-qplot(Petal.Width, Petal.Length, col=Species, data=training)
p + geom_point(aes(x=Petal.Width, y = Petal.Length, col = Species), size = 5, shape=4, data=irisP)
# predict new values
pred<-predict(modFit, testing)
testing$predRight<-pred==testing$Species
table(pred, testing$Species)
qplot(Petal.Width, Petal.Length, colour=predRight, data=testing, main="new data Predicting")

# Lecture 22- Boosting ------------------

# Lecture 23- Model Based Prediction -----