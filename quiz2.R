# Quiz #2

# Question 1 ----
# Split data into testing and training proprotionally by approx 50%
library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)
adData <- data.frame(diagnosis,predictors)
testIndex <- createDataPartition(diagnosis, p = 0.50,list=FALSE)
training <- adData[-testIndex,]
testing <- adData[testIndex,]

# Question 2 ------
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain <- createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training <- mixtures[ inTrain,]
testing <- mixtures[-inTrain,]
summary(training)
# make feature plot
xnames <- colnames(concrete)[1:8]
featurePlot(x=training[, xnames], y=training$CompressiveStrength, plot="pairs")
# make histogram
hist(training$Superplasticizer, main="", xlab="Superplasticizer")
# show SuperPlasticizer variable is skewed
plot(x = training$Superplasticizer, y = training$CompressiveStrength)
# why is a log tranform of SuperPlasticizer a poor choice for a variable?
plot(x=log10(training$Superplasticizer+1),y=training$CompressiveStrength)
# Answer: There are a large number of values that are the same and even if you took 
# the log(SuperPlasticizer + 1) they would still all be identical so the 
# distribution would not be symmetric.

# Question 3 -------
# Find all the predictor variables in the training set that begin with IL. 
# Perform principal components on these variables with the preProcess() 
# function from the caret package. Calculate the number of principal components 
# needed to capture 90% of the variance. How many are there?
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData <-data.frame(diagnosis,predictors)
inTrain <- createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training <- adData[ inTrain,]
testing <- adData[-inTrain,]

# find columnts that begin with IL
colnames(training)
# columns 58 - 69 begin with IL

# find correlations between IL vrs
names(training)[c(58:69)]
M<-abs(cor(training[,c(58:69)]))
diag(M)<-0
which(M > 0.7, arr.ind = T)
# IL_3 and IL_16 more than 70% correlated

# PCA with caret package
preProc<-preProcess(training[,c(58:69)], method = "pca",thresh = 0.9, outcome = training$diagnosis)
preProc$rotation
# Answer = 9

# Question 4 --------
# Create a training data set consisting of only the predictors 
# with variable names beginning with IL and the diagnosis. 
# Build two predictive models, one using the predictors as they 
# are and one using PCA with principal components explaining 
# 80% of the variance in the predictors. Use method="glm" in the 
# train function. What is the accuracy of each method in the test set? 
# Which is more accurate?
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData <- data.frame(diagnosis,predictors)
adData2 <-adData[,c(1,58:69)]
inTrain <-createDataPartition(adData2$diagnosis, p = 3/4)[[1]]
training<-adData2[ inTrain,]
testing <-adData2[-inTrain,]

# Method 1 - Non PCA 
set.seed(32343)
modelFit<-train(diagnosis ~., data = training, method = "glm")
modelFit
modelFit$finalModel
predictions<-predict(modelFit,newdata=testing)
c1<-confusionMatrix(predictions,testing$diagnosis)
c1$overall[1] #Accuracy = 0.6463

# Method 2 - PCA 80% variance explained
modelFit <- train(training$diagnosis ~ ., 
                  method="glm", 
                  preProcess="pca", 
                  data=training, 
                  trControl=trainControl(preProcOptions=list(thresh=0.8)))
c2 <- confusionMatrix(testing$diagnosis, predict(modelFit, testing))
c2$overall[1] #Accuracy = 0.7196
