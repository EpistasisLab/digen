import argparse
import pickle

from digen import Benchmark

# Load a package with DIGEN benchmark
benchmark = Benchmark()

# seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
                    required=False, nargs='?')
args = parser.parse_args()

datasets = args.dataset
if args.dataset is None:
    datasets = benchmark.list_datasets()

# Create your default class here or import from the package. As an example, we re benchmarking ExtraTreesClassifier from scikit-learn:
from sklearn.neural_network import MLPClassifier

est = MLPClassifier()


# In order to properly benchmark a method, we need to define its parameters and their values.
# Please set the expected range of hyper parameters for your method below. For details, please refer to Optuna.
def params_myParamScope(trial):

    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_neurons_{i}', 4, 128))

    params = {
        'activation':  trial.suggest_categorical(name='activation', choices=['identity', 'logistic', 'tanh', 'relu']),
        'solver':  trial.suggest_categorical(name='solver', choices=['lbfgs', 'sgd', 'adam']),
        'alpha':  trial.suggest_loguniform('alpha', 0.0001, 1.0),
        'hidden_layer_sizes':  tuple(layers),
        'max_iter' : 10000
    }

    return params



# Perform optimization of the method on DIGEN datasets
results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
                             storage='sqlite:///'+datasets+'-mlp.db', local_cache_dir='.')
pickle.dump( results, open( datasets+".pkl", "wb" ) )