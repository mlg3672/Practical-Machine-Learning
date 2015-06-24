# Week 2 Notes and Code from Video Lectures
# Lecture 10 - Caret package ------------------------------
## predict function syntax
# obj/class - Package - function syntax
#lda - MASS           - predict(obj) 
#glm - stats          - predict(obj,type="response")
#gbm - gbm            - predict(obj,type="response", n.trees)
#mda - mda            - predict(obj,type="posterior")
#rpart - rpart        - predict(obj,type="prob")
#Weka - RWeka         - predict(obj,type="probability")
#LogitBoost - CaTools - predict(obj,type="raw", nIter)

# data splitting
library(caret)
library(kernlab)
data(spam)
inTrain<-createDataPartition(y=spam$type,p=0.75,list=FALSE)
training<-spam[inTrain,]
testing<-spam[-inTrain,]
dim(training)

# fit general linear model to data
set.seed(32343)
modelFit<-train(type ~., data = training, method = "glm")
modelFit
modelFit$finalModel
predictions<-predict(modelFit,newdata=testing)
confusionMatrix(predictions,testing$type)

# Lecture 11 - Data Slicing --------------------------------
library(caret)
library(kernlab)
data(spam)
folds<-createFolds(y=spam$type, k = 10,list=TRUE, returnTrain=TRUE)
#note: if returnTRAIN=FALSE return only test sets
#what is the length of each fold?
sapply(folds,length)
#which samples are in each fold?
folds[[3]][1:10]
folds<-createResample(y = spam$type,times = 10, list = TRUE)
folds<-createTimeSlices(y = time, initialWindow = 20, horizon = 10)
names(folds)

# Lecture 12 - Training options --------------------------------------
modelFit<-train(type ~.,data = training, method = "glm")
args(train.default)
args(trainControl)

# Lecture 13 - Plotting Predictors ----------------------------------
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
summary(Wage)
inTrain<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
training<-Wage[inTrain,]
testing<-Wage[-inTrain,]
dim(training);dim(testing)
featurePlot(x = training, c("age","education","jobclass"),y = training$wage,plot = "pairs")
qplot(age, wage, colour = jobclass, data = training)

# apply linear model to visualization
qq<-qplot(age, wage, colour = education, data = training)
qq + geom_smooth(method = 'lm', formula = y~x)

# break up dataset into factors based on quantile groups
library(Hmisc)
cutWage<-cut2(training$wage, g = 3)
table(cutWage)
p1<-qplot(cutWage, age, data = training, fill = cutWage, geom = c("boxplot"))
p1

# add on top of boxplot points themselves
p2<-qplot(cutWage, age, data = training, fill = cutWage, geom = c("boxplot","jitter"))
# makes two plots side by side
require(grid)
require(gridExtra)
grid.arrange(p1,p2,ncol=2)

# view tables of data
t1<-table(cutWage, training$jobclass)
t1

# get the proportions in each row=1 or col=2
prop.table(t1,1)

# density plot for continuous predictors
qplot(wage, colour = education, data = training, geom="density")

# Lecture 14 - Basic PreProcessing: Standardizing ---------------------------
library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=F)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)
hist(training$capitalAve, main="", xlab="ave. capital run length")
mean(training$capitalAve)
sd(training$capitalAve)
# Standardizing
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve)) / sd(trainCapAve)
mean(trainCapAveS)
sd(trainCapAveS)
# Standardizing testing
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve)) / sd(trainCapAve)
mean(testCapAveS)
sd(testCapAveS)

preObj <- preProcess(training[,-58], method=c("center", "scale"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)
testCapAveS <- predict(preObj, testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)
set.seed(32343)
#centering and scaling to remove strong bias and large variability
modelFit <- train(type~., data=training, 
                  preProcess=c("center", "scale"), method="glm")
modelFit
# standardizing - Box-cox transform
preObj <- preProcess(training[,-58], method=c("BoxCox"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1, 2))
hist(trainCapAveS)
qqnorm(trainCapAveS)
# standardizing - Imputing data
set.seed(13343)
training$capAve <- training$capitalAve
# make some NA values
selectNA <- rbinom(dim(training)[1], size=1, prob=0.05)==1
training$capAve[selectNA] <- NA
# impute and standardize
preObj <- preProcess(training[,-58], method="knnImpute")
library(RANN)
# predict values with NA
capAve <- predict(preObj, training[,-58])$capAve
# standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth - mean(capAveTruth)) / sd(capAveTruth)
# look at how close impute values are to NA values
quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA])
quantile((capAve - capAveTruth)[!selectNA])

# Lecture 15 - Covariate Creation -----------------------------------
## Level 1: raw data >>> predictors or covariates
## images - edges, corners, blobs, ridges
## webpages - number, type of images
## text files - frequency of words, phrases, capital letters

# dummy variables
data(Wage)
inTrain<-createDataPartition(y=Wage$wage,p=0.5,list=FALSE)
training<-Wage[inTrain,]
testing<-Wage[-inTrain,]
table(training$jobclass)
# make factor variable to binary
dummies<-dummyVars(wage~jobclass, data = training)
head(predict(dummies, newdata = training))

library(caret)
# detect variables with no variability
nsv<-nearZeroVar(training, saveMetrics = TRUE)
nsv
# note if zeroVar is TRUE no variability in data, in this case sex and region
library(splines)
# fit data to create polynomials
bsBasics<-bs(training$age,df=3)
# gives coefficients
bsBasics
lm1<-lm(wage~bsBasics, data = training)
plot(training$age,training$wage, pch=19,cex = 0.5)
points(training$age, predict(lm1, newdata = training), col = "red", pch = 19, cex = 0.5)
predict(bsBasics, age = testing$age)
### issue: how to predict values for test set from training set model??
lm2<-lm(wage~bs(testing$age,df=3),data = testing)
plot(testing$age,testing$wage, pch=19,cex = 0.5)
points(testing$age, predict(lm2, newdata = testing), col = "red", pch = 19, cex = 0.5)
## Level 2: transforming tidy covariates
# use preProcess in caret
# spline models use gam method

# Lecture 16 -  Preporocessing with Prinipal Component Analysis --------
## in the case where variables are highly correlated, summarize
## rather than use multiple similar variables
library(caret)
library(kernlab)
data(spam)
inTrain<-createDataPartition(y=spam$type,p=0.75,list=FALSE)
training<-spam[inTrain,]
testing<-spam[-inTrain,]
M<-abs(cor(training[,-58]))
diag(M)<-0
which(M > 0.8, arr.ind = T)
names(spam)[c(34,32)]
plot(spam[,34],spam[,32])

# svd matrix decomposition - requires scaling
smallSpam<-spam[,c(34,32)]
pComp<-prcomp(smallSpam)
plot(pComp$x[,1],pComp$x[,2])
pComp$rotation
typeColor<-((spam$type == "spam")*1 +1) 
pComp<-prcomp(log10(spam[,-58]+1))
plot(pComp$x[,1],pComp$x[,2],col = typeColor, xlab = "PC1", ylab = "PC2")

# PCA with caret package
preProc<-preProcess(log10(spam[,-58]+1), method = "pca",pcaComp = 2)
spamPC<- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1],spamPC[,2],col = typeColor)
trainPC<-predict(preProc,log10(training[,-58]+1))
modelFit<- train(training$type~.,methods = "glm", data = trainPC)
modelFit
modelFit$finalModel
testPC<-predict(preProc, log10(testing[,-58]+1))
confusionMatrix(testing$type, predict(modelFit, testPC))

# alternative - sets the number of PCs
modelFit<- train(training$type~.,method = "glm", preProcess = "pca", data = training)
modelFit
modelFit$finalModel
confusionMatrix(testing$type, predict(modelFit, testing))
predict(modelFit, newdata = testing)

# Lecture 17 - Predicting with Regression ---------
# key ideas:  (1) fit simiple regresssion model
#             (2) plug in new covariants
#             (3) for use with linear models
library(caret)
data("faithful")
set.seed(333)
inTrain<-createDataPartition(y=faithful$waiting,p=0.5,list=FALSE)
trainFaith<-faithful[inTrain,]
testFaith<-faithful[-inTrain,]
head(trainFaith)
plot(trainFaith$waiting, trainFaith$eruptions, pch = 19, col = "blue", 
     xlab = "Waiting", ylab = "Eruptions")
lm1<-lm(eruptions~waiting, data = trainFaith)
summary(lm1)
lines(trainFaith$waiting, lm1$fitted.values, lwd = 3)
coef(lm1)[1]+coef(lm1)[2]*80
newdata<-data.frame(waiting=80)
predict(lm1,newdata)
par(mfrow = c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions)
lines(trainFaith$waiting, lm1$fitted, lwd = 3)
plot(testFaith$waiting, testFaith$eruptions)
lines(testFaith$waiting, predict(lm1, newdata = testFaith), lwd = 3)

# calc RMSE on training to set errors
sqrt(sum((lm1$fitted.values-trainFaith$eruptions)^2))
# calc RMSE on test
sqrt(sum((predict(lm1,newdata = testFaith)-testFaith$eruptions)^2))
# note: RMSE of test usually higher than train

# prediction intervals for test
pred1<-predict(lm1, newdata = testFaith, interval = "prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting, testFaith$eruptions)
matlines(testFaith$waiting[ord],pred1[ord,],type= "l",col = c(1,2,2), lty = c(1,1,1), lwd = 3)
modFit<-train(eruptions~waiting, data = trainFaith, method = "lm")
summary(modFit$finalModel)

# Lecture 18 - Prediction Regression of Multiple Covariates -----
# which predictors are most important to include in a data model?
library(ISLR)
library(ggplot2)
library(caret)
data("Wage")
# select out logwage col in dataset
Wage<-subset(Wage, select =-c(logwage))
summary(Wage)
# split data to train and test sets
inTrain<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
training<-Wage[inTrain,]
testing<-Wage[-inTrain,]
dim(training); dim(testing)
# explore data
featurePlot(x=training[,c("age","education","jobclass")],y=training$wage,plot = "pairs")
qplot(age,wage,data=training)
# how to predict outliers? first color by variable to discover source of variability
qplot(age,wage,colour=jobclass,data=training)
qplot(age,wage,colour=education,data=training)

# fit linear model
# jobclass and education are indicator variables
modFit<-train(wage~age+jobclass+education, method="lm", data=training)
finMod<-modFit$finalModel
print(modFit)
# diagnostic plot, residuals = diff betweeen real and predicted values)
plot(finMod,1,pch=19,cex=0.5, col= "#00000010")
# residuals should be along zero on y axis
qplot(finMod$fitted, finMod$residuals, colour=race, data=training)
# plot by index i.e. row number
# a trend wrt row number suggests variable missing from model e.g. time, age var
plot(finMod$residuals, pch=19)
# ideally have straight line where model equal to predictions
pred<-predict(modFit, testing)
qplot(wage, pred, colour=year, data = testing)
# predict with all variables in data set
modFitAll<- train(wage~., method="lm")
pred<-predict(modFitAll, testing)
qplot(wage, pred, data=testing)
