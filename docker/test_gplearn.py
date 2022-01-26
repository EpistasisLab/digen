from digen import Benchmark
import argparse
from gplearn.genetic import SymbolicClassifier
import argparse

from gplearn.genetic import SymbolicClassifier

from digen import Benchmark

# Load a package with DIGEN benchmark
benchmark = Benchmark()

# seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
                    required=False, nargs='?')
args = parser.parse_args()

datasets = benchmark.list_datasets()
if args.dataset is not None:
    assert (args.dataset in datasets)
    datasets = args.dataset

# Create your default class here or import from the package. As an example, we re benchmarking ExtraTreesClassifier from scikit-learn:
est = SymbolicClassifier()


# In order to properly benchmark a method, we need to define its parameters and their values.
# Please set the expected range of hyper parameters for your method below. For details, please refer to Optuna.
def params_myParamScope(trial):
    population_size = trial.suggest_categorical(name='population_size', choices=[100, 250, 500, 1000])
    generations = trial.suggest_categorical(name='generations', choices=[int(100000 / population_size)])
    print(str(generations))
    params = {
        'tournament_size': trial.suggest_int('tournament_size', 10, 25),
        'population_size': population_size,
        'generations': generations,
        'p_crossover': trial.suggest_float('p_crossover', 0.1, 0.9, step=0.1)
    }
    return params


est = SymbolicClassifier(
    # tournament_size=20,
    # stopping_criteria=0.0,
    # const_range=(0, 1.0),
    # init_depth=(2, 6),
    init_method='half and half',
    # metric='mean absolute error',
    # parsimony_coefficient=0.001,
    # p_crossover=0.9,
    # p_subtree_mutation=0.01,
    # p_hoist_mutation=0.01,
    # p_point_mutation=0.01,
    # p_point_replace=0.05,
    # max_samples=1.0,
    function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt'),
    population_size=1000,
    generations=100
)

# Perform optimization of the method on DIGEN datasets
results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
                             storage='sqlite:///test.db', local_cache_dir='.')
