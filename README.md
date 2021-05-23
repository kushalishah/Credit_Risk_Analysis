# Credit_Risk_Analysis

## Overview of Project

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I used different techniques to train and evaluate models with unbalanced classes. I used `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversampled the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, I used a combinatorial approach of over and undersampling using the `SMOTEENN` algorithm. Next, I compared two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. I evaluated the overall performance of these models based on whether or not they should be used to predict credit risk.

## Results

### RandomOverSampler Model

![oversample](https://github.com/kushalishah/Credit_Risk_Analysis/blob/main/Images/oversampling.png)

The balanced accuracy score is 63%. The high_risk precision is about 1% with a 67% sensitivity which makes an F1 of 1%. Due to the high number of the low_risk population, its precision is almost 100%.

### SMOTE Model

![smote](https://github.com/kushalishah/Credit_Risk_Analysis/blob/main/Images/smote.png)

We can see that the results are similar to the model above. The balanced accuracy score is 62% with a 58% sensitivity. 

### ClusterCentroids Model

![undersampling](https://github.com/kushalishah/Credit_Risk_Analysis/blob/main/Images/undersampling.png)

The balanced accuracy score is 52% with a 60% sensitivity. 

### SMOTEENN Model

![combinationsampling](https://github.com/kushalishah/Credit_Risk_Analysis/blob/main/Images/combinationsampling.png)

The balanced accuracy score is 61% with a 67% sensitivity. Due to the high number of false positives, the low_risk sensitivity is 55%.

### BalancedRandomForestClassifier Model

![brfc](https://github.com/kushalishah/Credit_Risk_Analysis/blob/main/Images/brfc.png)

The balanced accuracy score is 91%.

### EasyEnsembleClassifier Model

![easyensemble](https://github.com/kushalishah/Credit_Risk_Analysis/blob/main/Images/easyensemble.png)

The accuracy is 92%, F1 is 12% and higher than any other model.

## Summary

This analysis is trying to find the best model that can detect if a loan is high risk or not. Becasue of that, we need to find a model that lets the least amount of high risk loans pass through undetected. That correlating statistic for this is the recall rate for high risk. Looking through the different models, the ones that scored the highest were: Easy Ensemble Classifying, SMOTEENN Sampling, Naive Random Oversampling.

While this is the most important statistic that is pulled from this analysis, another important statistic is recall rate for low risk as it shows how many low risk loans are flagged as high risk. Looking through the different models, the ones that scored the highest were:

1. Balanced Random Forest Classifying (92%)
2. Easy Ensemble Classifying (92%)

After taking these two statistics over the others, we can look at the accurary score to get a picture of how well the model performs in general. The models with the highest accuracy scores were:

1. Easy Ensemble Classify (92%)
2. SMOTEENN Sampling (61%)
3. Balanced Random Forest Classifying (92%)

After factoring in these three main statistics, the model that I would recommend to use for predicting high risk loans is the Easy Ensemble Classifying model.
