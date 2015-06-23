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
modFit<-train(species~., method="rpart", data = training)
print(modFit$finalModel)
plot(modFit$finalMOdel, uniform=TRUE, main= "Classification Tree")
text(modFit$finalModel, use.n= TRUE, all = TRUE, cex = 0.8)
library(rattle)
fancyRpartPlot(modFit$finalModel)
predict(modFit, newdata = testing)

# other tree building options
modFit<-train(species~., method="party", data = training)
modFit<-train(species~., method="tree", data = training)

# Lecture 20- Bagging -------------------
# Lecture 21- Random Forests ------------
# Lecture 22- Boosting ------------------
# Lecture 23- Model Based Prediction -----