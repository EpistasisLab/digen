# Frequently asked questions (FAQ)

#### I got a different AUROC value for one of the methods than the one in DIGEN. Is it a bug?

No, it's not a bug.
It is possible that the specific settings of ML method (even default ones) are better than the results obtained in hyper-parameters tuning process performed with Optuna.
DIGEN does up to 200 hyper-parameter optimizations for each ML method. 
Although this is a decent number of trials, this does not guarantee that the results will be optimal, unless a Docker image is used.


#### Are the results from DIGEN reproducible?

Yes! 

In order to provide full reproducibility of the experiment across different platforms, we have included a Docker configuration files, allowing to build a container with specific libraries and configurations.

Specifically, the following versions of the packages were used to assure reproducibility:

    numpy 1.19.5
    sklearn 0.22.2.post1
    xgboost 1.3.1
    lightgbm 3.1.1
    optuna 2.4.0
    pandas==1.1.5
    numpy==1.19.5
    optuna==2.4.0
    deap==1.3
 

