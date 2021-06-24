# What is DIGEN?

Diverse and Generative ML benchmark (DIGEN) is a modern machine learning benchmark, which includes:
- 40 datasets in tabular numeric format specially designed to differentiate the performance of some of the leading Machine Learning (ML) methods, and
- a package to perform reproducible benchmarking that simplifies comparison of performance of the methods.

DIGEN provides comprehensive information on the datasets, including:
- ground truth - a mathematical formula presenting how the endpoint was generated for each of the datasets
- the results of exploratory analysis, which includes feature correlation and histogram showing how binary endpoint was calculated.
- multiple statistics on the datasets, including the AUROC, AUPRC and F1 scores
- each dataset comes with Reveiver-Operating Characteristics (ROC) and Precision-Recall (PRC) charts for tuned ML methods, 
- a boxplot with projected performance of the leading methods after hyper-parameter tuning (100 runs of each method started with different random seed)

Apart from providing a collection of datasets and tuned ML methods, DIGEN provides tools to easily tune and optimize parameters of any novel ML method, as well as visualize its performance in comparison with the leading ones.


# Using DIGEN

Apart from the datasets, DIGEN provides a comprehensive toolbox for analyzing the performance of a chosen ML method.
DIGEN uses [Optuna](https://github.com/optuna/optuna), a state of the art framework for optimizing hyper-parameters 

Please refer to our online documentation at [https://epistasislab.github.io/digen](https://epistasislab.github.io/digen)


# Tutorials
[DIGEN Tutorial](https://github.com/EpistasisLab/digen/blob/main/DIGEN%20Tutorial.ipynb) is a great place to start
For advanced use, e.g. customization, manipulations with the charts, additional statistics on the collection, please check our [Advanced Tutorial](https://github.com/EpistasisLab/digen/blob/main/DIGEN%20Advanced.ipynb).


# Included ML classifiers:

The following methods were included in our benchmark:
- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors
- LightGBM
- Logistic Regression
- Random Forest
- SVC
- XGBoost

