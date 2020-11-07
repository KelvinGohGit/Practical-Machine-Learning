---
title: "Project Course Practical Machine Learning"
author: "Kelvin"
date: "11/6/2020"
output: 
  html_document:
    keep_md: true
    self_contained: true

---



Background
=======================	
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Data
====
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.



### Loading library package


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
```

```
## Warning: package 'RColorBrewer' was built under R version 4.0.3
```

```r
library(rattle)
```

```
## Loading required package: tibble
```

```
## Loading required package: bitops
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 4.0.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 4.0.3
```

```
## corrplot 0.84 loaded
```

```r
library(corrr)
```

```
## Warning: package 'corrr' was built under R version 4.0.3
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 4.0.3
```

```
## Loaded gbm 2.1.8
```

```r
library(knitr)
```


### Loading dataset


```r
###loading the dataset
TrainingData<-read.csv("./project_data/pml-training.csv")
ValidationData<-read.csv("./project_data/pml-testing.csv")

dim(TrainingData)
```

```
## [1] 19622   160
```

```r
dim(ValidationData)
```

```
## [1]  20 160
```

Preprocessing and Data Preparation
================================

### Cleaning Data

Remove missing value data for training data and validation data

```r
TrainData<-TrainingData[,colSums(is.na(TrainingData))==0]
ValidData<-ValidationData[,colSums(is.na(ValidationData))==0]
dim(TrainData)
```

```
## [1] 19622    93
```

```r
dim(ValidData)
```

```
## [1] 20 60
```

```r
TrainData<-TrainData[,-c(1:7)]
ValidData<-ValidData[,-c(1:7)]
dim(TrainData)
```

```
## [1] 19622    86
```

```r
dim(ValidData)
```

```
## [1] 20 53
```

### Data Preparation for Prediction

*Split TrainData into TrainSet and TestSet*

Preparing the data for prediction by splitting the training data into 70% as train data and 30% as test data.  

The test data downloaded from the source will be used to validate the final prediction model
 

```r
set.seed(1234)
inTrain <- createDataPartition(TrainData$classe, p=0.7,list = FALSE)

TrainSet<-TrainData[inTrain,]
TestSet<-TrainData[-inTrain,]
dim(TrainSet)
```

```
## [1] 13737    86
```

```r
dim(TestSet)
```

```
## [1] 5885   86
```

*Remove variables with near zero variance*


```r
NZV<-nearZeroVar(TrainData)
TrainSet<-TrainSet[,-NZV]
TestSet<-TestSet[,-NZV]
dim(TrainSet)
```

```
## [1] 13737    53
```

```r
dim(TestSet)
```

```
## [1] 5885   53
```

After this cleaning, they are a total of 53 variables can be used to buid the prediction model


### Building Prediction Model

For this project,3 of the following algorithms will be used to build the prediction model and final model will be selected based on the level of accuracy.

 1) Random forecast
 2) Decision Tree
 3) Generalized Boosted Model(GBM)

### A. Random Forest

```r
#Model Fit
set.seed(1234)
controlRF<-trainControl(method="cv",number=3, verboseIter=FALSE)
modFitRF<-train(classe~., data=TrainSet,method="rf",trControl=controlRF)
modFitRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.69%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    2    0    0    1 0.0007680492
## B   16 2635    6    1    0 0.0086531226
## C    0   17 2369   10    0 0.0112687813
## D    0    1   26 2223    2 0.0128774423
## E    0    2    5    6 2512 0.0051485149
```

Test the model modFitRF on Testset dataset to check on Accuracy of the prediction 



```r
# Prediction on Test dataset
predictRF<-predict(modFitRF,newdata=TestSet)
ConfMatRF<-confusionMatrix(predictRF,as.factor(TestSet$classe))
ConfMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    5    0    0    0
##          B    1 1129   13    1    0
##          C    1    5 1011    9    1
##          D    0    0    2  953    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.991, 0.9953)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9912   0.9854   0.9886   0.9991
## Specificity            0.9988   0.9968   0.9967   0.9996   0.9998
## Pos Pred Value         0.9970   0.9869   0.9844   0.9979   0.9991
## Neg Pred Value         0.9995   0.9979   0.9969   0.9978   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1918   0.1718   0.1619   0.1837
## Detection Prevalence   0.2850   0.1944   0.1745   0.1623   0.1839
## Balanced Accuracy      0.9988   0.9940   0.9910   0.9941   0.9994
```


![](Project-Course-RMD_files/figure-html/unnamed-chunk-8-1.png)<!-- -->
The accuracy rate using the random forest is very high: Accuracy level is 0.9934 and therefore the out-of-sample-error is equal to 0.69%.  

### B. Decision Tree

```r
#model fit
set.seed(1234)
modFitDT<-rpart(classe~.,data=TrainSet,method="class")
fancyRpartPlot(modFitDT)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![](Project-Course-RMD_files/figure-html/unnamed-chunk-9-1.png)<!-- -->


```r
# Prediction on TestSet dataset
predictDT<-predict(modFitDT,newdata=TestSet, type="class")
confMatDT<-confusionMatrix(predictDT,as.factor(TestSet$classe))
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1522  167   12   49   13
##          B   58  706  100   79   96
##          C   47  109  819  148  139
##          D   25   94   67  609   52
##          E   22   63   28   79  782
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7541          
##                  95% CI : (0.7429, 0.7651)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6885          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9092   0.6198   0.7982   0.6317   0.7227
## Specificity            0.9428   0.9298   0.9088   0.9516   0.9600
## Pos Pred Value         0.8633   0.6795   0.6490   0.7190   0.8029
## Neg Pred Value         0.9631   0.9106   0.9552   0.9295   0.9389
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2586   0.1200   0.1392   0.1035   0.1329
## Detection Prevalence   0.2996   0.1766   0.2144   0.1439   0.1655
## Balanced Accuracy      0.9260   0.7748   0.8535   0.7917   0.8414
```


```r
#plot matrix result
plot(confMatDT$table,col=confMatDT$byclass,main=paste("Decision Tree-Accuracy", round(confMatDT$overall["Accuracy"],4)))
```

![](Project-Course-RMD_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

The accuracy rate of the model for decision tree is 0.7541


### C. Generalized Boosted Model (GBM)

```r
#model fit
set.seed(1234)
controlGBM<-trainControl(method="repeatedcv",number=5, repeats=1)
modFitGBM<-train(classe~., data=TrainSet,method="gbm",trControl=controlGBM, verbose=FALSE)
modFitGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 52 had non-zero influence.
```





```r
# Prediction on TestSet dataset
predictGBM<-predict(modFitGBM,newdata=TestSet)
confMatGBM<-confusionMatrix(predictGBM,as.factor(TestSet$classe))
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1653   27    0    3    1
##          B   10 1083   36    3   12
##          C    8   29  977   26   10
##          D    2    0   12  927    8
##          E    1    0    1    5 1051
## 
## Overall Statistics
##                                           
##                Accuracy : 0.967           
##                  95% CI : (0.9622, 0.9714)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9583          
##                                           
##  Mcnemar's Test P-Value : 2.194e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9875   0.9508   0.9522   0.9616   0.9713
## Specificity            0.9926   0.9871   0.9850   0.9955   0.9985
## Pos Pred Value         0.9816   0.9467   0.9305   0.9768   0.9934
## Neg Pred Value         0.9950   0.9882   0.9899   0.9925   0.9936
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2809   0.1840   0.1660   0.1575   0.1786
## Detection Prevalence   0.2862   0.1944   0.1784   0.1613   0.1798
## Balanced Accuracy      0.9900   0.9690   0.9686   0.9786   0.9849
```



```r
#plot matrix result
plot(confMatGBM$table,col=confMatGBM$byclass,main=paste("Generalized Boosted Model-Accuracy", round(confMatGBM$overall["Accuracy"],4)))
```

![](Project-Course-RMD_files/figure-html/unnamed-chunk-14-1.png)<!-- -->


### Applying Selected Model
===========================

Comparing all 3 prediction models, Random Forests gave highest Accuracy among the three.Hence,the model will be applied on the validation set of data


```r
# Accuracy for all 3 models
round(ConfMatRF$overall["Accuracy"],4)
```

```
## Accuracy 
##   0.9934
```

```r
round(confMatDT$overall["Accuracy"],4)
```

```
## Accuracy 
##   0.7541
```

```r
round(confMatGBM$overall["Accuracy"],4)
```

```
## Accuracy 
##    0.967
```

### Validation of Final Prediction Model 
=========================================

```r
predictvalid<-predict(modFitRF, newdata=ValidData)
predictvalid
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```










