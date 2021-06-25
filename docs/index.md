# What is DIGEN?

Diverse and Generative ML benchmark (DIGEN) is a modern machine learning benchmark, which includes 40 datasets specially designed to differentiate the performance of some of the leading Machine Learning (ML) methods.

DIGEN provides comprehensive information on the datasets, as well as on the performance of the leading ML methods.
For each of the datasets, we provide a **mathematical formula** for the endpoint (ground truth) as well as the results of exploratory analysis, which includes feature correlation and histogram showing how binary endpoint was calculated.

Each dataset comes also with Reveiver-Operating Characteristics (ROC), Precision-Recall (PRC) charts for tuned ML methods with 100 guided optimizations, as well as boxplot showing the expected performance of the methods tested on 100 different starting random seeds and with 100 optimizations for each random seed.


## Installation 
	
Installation of the following packages is required in order to use DIGEN:

	pandas>=1.05
	numpy>=1.19.5
	optuna>=2.4.0
	scikit-learn>=0.22.2
	importlib_resources


In order to reproduce the computations, the following packages are also suggested:

	deap>=1.3
	digen
	seaborn
	Matplotlib
	lightgbm>=3.1.1
	xgboost>=1.3.2
	requests
	argparse
	inspect



## How to install DIGEN?

Download the latest release of DIGEN from PyPI:

	pip install digen


## Citing DIGEN

DIGEN was developed by Patryk Orzechowski and Jason H. Moore at the University of Pennsylvania.
