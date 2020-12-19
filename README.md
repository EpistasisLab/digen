# What is DIGEN?

Diverse and Generative ML benchmark (DIGEN) is a modern machine learning benchmark, which includes 40 datasets specially designed to differentiate the performance of some of the leading Machine Learning (ML) methods.
DIGEN provides comprehensive information on the datasets, as well as on the performance of the leading ML methods.
For each of the datasets, we provide a mathematical formula for the endpoint (ground truth) as well as the results of exploratory analysis, which includes feature correlation and histogram showing how binary endpoint was calculated.
Each dataset comes also with Reveiver-Operating Characteristics (ROC) and Precision-Recall (PRC) charts for tuned ML methods, as well as boxplot showing their expected performance with 100 random initializations of the datasets with the same data model.


# Using DIGEN

Apart from the datasets, DIGEN provides a comprehensive toolbox for analyzing the performance of a chosen ML method.
DIGEN uses[Optuna](!https://github.com/optuna/optuna), a state of the art framework for optimizing hyper-parameters 

More to follow...

# Benchmarked ML classifiers:

The following methods were benchmarked against DIGEN.
- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors
- LightGBM
- Logistic Regression
- Random Forest
- SVC
- XGBoost

