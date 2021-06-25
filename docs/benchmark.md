# How to use DIGEN?
The most common usecase of using DIGEN is validating performance of a new method against the benchmark. 

In order to use DIGEN, we first import the package:
    
    from digen import Benchmark
    benchmark = Benchmark()
    
The most common use case includes validation of a new method against DIGEN.
First, we need to define our new classifier. For this example, we will use ExtraTreesClassifier from scikit-learn:

    from sklearn.ensemble import ExtraTreesClassifier
    est=ExtraTreesClassifier()


In order to benchmark a method, we need to define its parameters and their values.
Please set the expected range of hyper parameters for your method below. For details, please refer to [Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)

    def params_myParamScope(trial):
        params={
            'n_estimators' : trial.suggest_int('n_estimators',10,100),
            'criterion' : trial.suggest_categorical(name='criterion',choices=['gini', 'entropy']),
            'max_depth' : trial.suggest_int('max_depth', 1, 10),
        }
        return params


After defining distributions and scope of each of the hyperparameters, all we need to do is to perform optimizations 
on each of the DIGEN datasets in order to fairly compare its performance against predefined methods:

    results=benchmark.optimize(est=est, 
                               parameter_scopes=params_myParamScope, 
                               storage='sqlite:///test.db',
                               local_cache_dir='.')

This command will initiate a benchmarking experiment with 200 optimizations per each of the datasets (it may take a while!).
It is also possible to perform computations on a single dataset, e.g.:

    results=benchmark.optimize(est=est, 
                               dataset='digen8_4426',
                               parameter_scopes=params_myParamScope, 
                               storage='sqlite:///test.db',
                               local_cache_dir='.')



DIGEN provides a comprehensive API with multiple visualizations, selections and full access and logging of the performed operations.
For the reference, please check our [Tutorial](https://github.com/EpistasisLab/digen/blob/main/DIGEN%20Tutorial.ipynb)
[Advanced Tutorial](https://github.com/EpistasisLab/digen/blob/main/DIGEN%20Advance.ipynb).

## Naming convention 
Apart from numeric identifiers of the datasets, we are also using the following abbreviations for ML methods:

* X - XGBoost,
* L - LightGBM, 
* G - Gradient Boosting, 
* F - Random Forest, 
* S - Support Vector Classifier, 
* K - K-Nnearest Neighbors,
* R - Logistic Regression, 
* D - Decision Tree.

The abbreviations used in DIGEN reflect the performance of tuned ML methods on a dataset created with a specific seed. The name of each dataset  starts with letters determining the order of the performances of the tuned ML methods included in the experiment. If two or more ML methods have the same score, a hyphen is used to separate them. 
Multiple datasets may have the same order of the methods.




## Storing ML experiments

We have adapted the following strategy to ensure the unique
identification of the experiments. Each mathematical formula was
represented as a string extracted from DEAP and was hashed using an MD5
algorithm.

The experiments are stored as studies in a SQLite dataset as Optuna storage (SQLite) in the
following format *digenNUM_SEED_method*, where *NUM* stands for a dataset it,
*SEED* - a random seed used as an initializer of the dataset and the methods, 
and *method* is a classifier that was used to analyze a dataset.
In the supporting files within the package, some additional properties ma be found, such as
*methodUP* and *methodDOWN*, which are two methods used to compete against each other, and *hash* that represents an MD5 shortcut of a mathematical formula. 
Hashing an equation doesn't guarantee a completely unique mathematical
formula (e.g. swapping the operands in sum creates a different hash
function, so as adding a meaningless addition of 0, or multiplication by
1). Nonetheless, it allows to identify redundancy and do not repeat steps that were already explored.

