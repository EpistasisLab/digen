import sys
import numpy as np
import pandas as pd
import re
import random
import itertools
import operator
import argparse
import inspect

from deap import base, tools, gp, creator
from digen import Benchmark, defaults
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


estimator_map={
    'GradientBoostingClassifier' : GradientBoostingClassifier(),
    'LGBMClassifier' : LGBMClassifier(),
    'XGBClassifier' : XGBClassifier(),
    'DecisionTreeClassifier' : DecisionTreeClassifier(),
    'LogisticRegression' : LogisticRegression(),
    'KNeighborsClassifier' : KNeighborsClassifier(),
    'RandomForestClassifier' :  RandomForestClassifier(),
    'SVC' : SVC(),
}


print(inspect.getsource(defaults.params_XGBClassifier))


# Load a package with DIGEN benchmark
benchmark=Benchmark()

#seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)", required=False, nargs='?')
args = parser.parse_args()

datasets=benchmark.list_datasets()
if args.dataset is not None:
    assert(args.dataset in datasets)
    datasets=args.dataset


#List the estimators to run
estimators = ['XGBClassifier','LGBMClassifier']

#Optimize methods
for name, est in estimator_map.items():
    print(str(inspect.getsource(eval('defaults.params_'+name))))
    results=benchmark.optimize(est=est,datasets=datasets, parameter_scopes='defaults.params_'+name, storage='sqlite:///test.db',local_cache_dir='.')

