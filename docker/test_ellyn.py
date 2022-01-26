#! /usr/bin/env python3

from digen import Benchmark
import argparse
from ellyn import ellyn

# Load a package with DIGEN benchmark
benchmark = Benchmark(n_trials=5)

# seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
    required=False,
    nargs='?')
args = parser.parse_args()

datasets = benchmark.list_datasets()
if args.dataset is not None:
    assert (args.dataset in datasets)
    datasets = args.dataset

est = ellyn(classification=True, class_m4gp=True)


def params_myParamScope(trial):
    params = {
        'g': trial.suggest_int('g', 5, 20),
        'selection': trial.suggest_categorical(name='selection', choices=['lexicase', 'tournament']),
        'popsize': trial.suggest_int('popsize', 25, 500),
    }
    return params


# example dataset - digen8_4426

results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
                             storage='sqlite:///test.db', local_cache_dir='.')
