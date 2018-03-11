# SECOM_Prediction

Analysis of data from a semi-conductor manufacturing process using different classification models and Principal Component Analysis

Predictive Performance analysis of a Semiconductor Manufacturing dataset (SECOM) taken from UCI Machine Learning Repository website. My objective is to check whether a component will pass or fail based on 561 attributes (i.e. variables collected from sensors and or process measurement points).

To filter the most importat variables PCA is used. For prediction various machine learning algorithms have been implemented and there performance have been validated through cross-validation. Algorithms implemented: Logistic Regression, LDA, QDA, KNN, tree / tree pruning, SVM, Random Forests with bagging and boosting