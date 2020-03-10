# ****************************************************
# Ruoshi Li
# Date: Mar 7th, 2020
# Comments: Practical Machine Learning - Final Project
# ****************************************************

# Clear all variables and prior sessions
rm(list = ls(all = TRUE))

# Set 'working directory'
wdir <- "/Users/Ruoshi/Documents/Study/Data Science/Coursera_Johns_Hopkins/Class 8/Final Project"
setwd(wdir)

# load necessary libraries
library(caret)
library(kernlab)
library(randomForest)
library(rattle)

# import training and testing datasets
training_data <- read.csv("./pml-training.csv", header = T)
testing_data <- read.csv("./pml-testing.csv", header = T)
# check dataset dimensions
dim(training_data)
dim(testing_data)

# remove nzv variables
nzv <- nearZeroVar(training_data)
training_data <- training_data[ ,-nzv]
testing_data <- testing_data[ ,-nzv]
dim(training_data)
dim(testing_data)
# remove cols that more than 90% of their values are NAs from training dataset
training_remove <- which(colSums(is.na(training_data) |training_data=="")>0.9*dim(training_data)[1]) 
training_data <- training_data[ , -training_remove]
testing_data <- testing_data[ , -training_remove]
training_data <- training_data[ , -c(1:6)]
testing_data <- testing_data[ , -c(1:6)]
dim(training_data)
dim(testing_data)
# split training dataset into training and testing parts
set.seed(123)
inTrain <- createDataPartition(training_data$classe, p = 0.7, list = FALSE)
training <- training_data[inTrain, ]
testing <- training_data[-inTrain, ]
dim(training)
dim(testing)
str(training)

# prediction using classfication tree
set.seed(1234)
fitControl <- trainControl(method = "cv", number = 5, verboseIter = F)
mod_tree <- train(classe ~ ., data = training, method = "rpart", trControl = fitControl)
mod_tree$finalModel
suppressMessages(library(rattle))
fancyRpartPlot(mod_tree$finalModel)
tree_pred <- predict(mod_tree, newdata = testing)
tree_confMatrix <- confusionMatrix(testing$classe, tree_pred)
tree_confMatrix

# prediction using gbm
mod_gbm <- train(classe ~ ., data = training, method = "gbm", trControl = fitControl)
mod_gbm$finalModel
plot(mod_gbm)
gbm_pred <- predict(mod_gbm, newdata = testing)
gbm_confMatrix <- confusionMatrix(testing$classe, gbm_pred)
gbm_confMatrix
# prediction using random forest
mod_rf <- train(classe ~ ., data = training, method = "rf", trControl = fitControl)
mod_rf$finalModel
plot(mod_rf)
rf_pred <- predict(mod_rf, newdata = testing)
rf_confMatrix <- confusionMatrix(testing$classe, rf_pred)
rf_confMatrix

# Prediction using testing dataset
test_pred <- predict(mod_rf, newdata = testing_data)
test_pred