# Quiz 3 

library(caret)
library(AppliedPredictiveModeling)
library(pgmm)
library(rpart)
# Question 1 ----------
data("segmentationOriginal")
# 1. Subset the data to a training set and testing set based on the Case variable in the data set. 

# 2. Set the seed to 125 and fit a CART model with the rpart method using 

#     all predictor variables and default caret settings. 
# 3. In the final model what would be the final model prediction for 
#    cases with the following variable values:
#   a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 
#   b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
#   c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
#   d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2

# Answer: a. PS b. WS c. PS d. Not Possible to Predict
# Question 2 ----------
# Question 3 ----------
# Question 4 ----------