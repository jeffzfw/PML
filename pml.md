---
title: "PML"
output: html_document
---
## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement

a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Given data from accelerometers, the goal is to predict the class of action which is one of the following.

1.exactly according to the specification A
2.throwing elbows to the front B
3.lifting the dumbbell only halfway C
4.lowering the dumbbell only halfway D
5.throwing the hips to the front E.

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Loading data

```r
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",mode = "wb",destfile = "./train.csv")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",mode = "wb",destfile = "./test.csv")
train <- read.csv("./train.csv")
test <- read.csv("./test.csv")
```
## Cleaning data
Check the train data,found lot of NAs.
Decide to drop the useless variables to make clean data.

```r
colsOmit <-  grep("^var|^avg|^max|^min|^std|^amplitude",names(train))
trainF <- train[-c(colsOmit)]
varsO <- names(trainF) %in% c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
trainF  <- trainF[!varsO]
```
we can see the predictor variables we still have:

```r
dim(trainF)[2]
```

```
## [1] 77
```
Decide to perform  zero and near-zero-variance analysis on data.

```r
require(caret)
require(Hmisc)
set.seed(12345)
nzv <- nearZeroVar(trainF,saveMetrics=FALSE)
trainF <- trainF[-c(nzv)]
```
The predictor variables remain:

```r
dim(trainF)[2]
```

```
## [1] 53
```
## Perform a correlation analysis on predicator

```r
trainFOcls <- trainF[-c(dim(trainF))]
colexcl <- findCorrelation(cor(trainFOcls), cutoff= 0.75)
trainF <- trainF[-c(colexcl)]
```
now we have these predictors:

```r
colnames(trainF)
```

```
##  [1] "yaw_belt"             "gyros_belt_x"         "gyros_belt_y"        
##  [4] "gyros_belt_z"         "magnet_belt_x"        "magnet_belt_z"       
##  [7] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [10] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_z"         
## [13] "magnet_arm_x"         "magnet_arm_z"         "roll_dumbbell"       
## [16] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [19] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "magnet_dumbbell_z"   
## [22] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [25] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [28] "accel_forearm_x"      "accel_forearm_z"      "magnet_forearm_x"    
## [31] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```
## Process the testing data

```r
trainFOcls <- trainF[-c(dim(trainF))]
testF <- test[colnames(trainFOcls)]
```
## Train a prediction model
divide train dataset into training(75%) and probe(25%) datasets.

```r
partitionF = createDataPartition(trainF$classe, p=0.75, list=F)
trainD <- trainF[partitionF,]
probeD <- trainF[-partitionF,]
```
using caret train fuction rf(random forest)algorithm.

```r
# using parallel mode to save time.
require(parallel)
require(doParallel)
trainD[-dim(trainD)]->X
preProc <- preProcess(X)
XCS <- predict(preProc, X)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
ctrl <- trainControl(classProbs=TRUE,savePredictions=TRUE,allowParallel=TRUE)
trainingModel <- train(classe ~ ., data=trainD, method="rf")

trainingModel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.69%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4180    3    0    0    2 0.001194743
## B    9 2827   12    0    0 0.007373596
## C    0   16 2540   11    0 0.010518115
## D    0    0   37 2371    4 0.016998342
## E    0    0    2    6 2698 0.002956393
```

## Testing training Model on probe data
Test the trainingModel on probeD

```r
testD <- predict(trainingModel,probeD)
confusionMatrix(testD, probeD[, "classe"])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    3    0    0    0
##          B    0  943    4    0    0
##          C    0    3  849    1    0
##          D    0    0    2  803    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9955, 0.9986)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9937   0.9930   0.9988   1.0000
## Specificity            0.9991   0.9990   0.9990   0.9995   1.0000
## Pos Pred Value         0.9979   0.9958   0.9953   0.9975   1.0000
## Neg Pred Value         1.0000   0.9985   0.9985   0.9998   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1923   0.1731   0.1637   0.1837
## Detection Prevalence   0.2851   0.1931   0.1739   0.1642   0.1837
## Balanced Accuracy      0.9996   0.9963   0.9960   0.9991   1.0000
```
The confusion matrix for prediction shows the Model is acceptable.

## save the final model

```r
trainingModel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.69%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4180    3    0    0    2 0.001194743
## B    9 2827   12    0    0 0.007373596
## C    0   16 2540   11    0 0.010518115
## D    0    0   37 2371    4 0.016998342
## E    0    0    2    6 2698 0.002956393
```

```r
save(trainingModel,file = "traingModel.rData")
```
The erro rate is less than 1%

## Do predict on test data
Load the traingModel.

```r
load(file="traingModel.rData", verbose=TRUE)
```

```
## Loading objects:
##   trainingModel
```
Apply trainingMOdel on test dataset.

```r
predictF = predict(trainingModel, newdata = testF)
predictF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Write the Prediction to files

```r
# Function to write a vector to files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_", i ,".txt")
    write.table(x[i], file = filename, quote = FALSE,
                row.names = FALSE, col.names = FALSE)
  }
}
# Call the function
pml_write_files(predictF)
```
