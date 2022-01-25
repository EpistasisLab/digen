import argparse
import itertools
import operator
import random

import numpy as np
import pandas as pd
from deap import base, tools, gp, creator

from digen import Benchmark


def get_random_data(rows, cols, seed=100):
    """ return randomly generated data is shape passed in """
    if seed is None:
        np.random.seed(seed)
        random.seed(seed)
    #    data = np.random.randint(0,3,size=(rows,cols))
    data = np.random.normal(size=(rows, cols))
    return pd.DataFrame(data)


def reclassify(y, threshold):
    # y - endpoint to reclassify to 0/1
    # threshold (0,1) - percentage of 1's in the reclassified endpoint

    order = np.argsort(y)
    new_y = np.zeros(len(y))
    new_y[order[int(len(y) * threshold):]] = 1
    return new_y.astype(int)


def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


benchmark = Benchmark()
# assign id in DIGEN to default seed
seedmap = dict(map(lambda x: (x.split('_')[0], x.split('_')[1]), benchmark.list_datasets()))
# assign id in DIGEN to equation
datamap = dict(map(lambda x, y: (x.split('_')[0], y), benchmark.get_models().keys(), benchmark.get_models().values()))

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
                    required=False, nargs='?')
parser.add_argument("-r", "--rows", default=1000, help="Number of rows", required=False, nargs='?')
parser.add_argument("-c", "--columns", default=10, help="Number of columns", required=False, nargs='?')
parser.add_argument("-s", "--seed", default=None, help="Random seed", required=False, nargs='?')

args = parser.parse_args()

if args.dataset:
    datamap = {args.dataset.split('_')[0]: datamap[args.dataset.split('_')[0]] for keys in datamap.keys()}
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, int(args.columns)), float, "X")

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(safeDiv, [float, float], float)

pset.addPrimitive(operator.eq, [float, float], float)
pset.addPrimitive(operator.ne, [float, float], float)
pset.addPrimitive(operator.ge, [float, float], float)
pset.addPrimitive(operator.gt, [float, float], float)
pset.addPrimitive(operator.le, [float, float], float)
pset.addPrimitive(operator.lt, [float, float], float)

pset.addPrimitive(min, [float, float], float)
pset.addPrimitive(max, [float, float], float)

randval = "rand" + str(random.random())[2:]
pset.addEphemeralConstant(randval, lambda: random.random() * 100, float)
pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # Returns a tuple of one tree.
ref_points = tools.uniform_reference_points(nobj=2, p=12)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
toolbox.register("map", map)  # Overload the map function

for key, equation in datamap.items():

    if args.seed:
        seed = int(args.seed)
    else:
        seed = int(seedmap[key])

    random.seed(seed)
    np.random.seed(seed)
    X = get_random_data(int(args.rows), int(args.columns), seed)

    individual = gp.PrimitiveTree.from_string(equation, pset)
    func = toolbox.compile(expr=individual)
    y = np.zeros(int(args.rows))

    # Dimensionality reduction.
    for j in range(X.shape[0]):  # For every row. Every row is passed through a tree.
        y[j] = func(*X.iloc[j]) if isinstance(X, pd.DataFrame) else func(*X[j, :])  # X
    org_y = y
    y = reclassify(y, 0.5)
    X = get_random_data(int(args.rows), int(args.columns), seed)

    X.columns = list(map(lambda k: 'X' + str(k), range(X.shape[1])))
    X['target'] = y
    filename = key + '_' + str(seed)
    X.to_csv(filename + '.tsv', index=None, compression='gzip', sep='\t')
    X.to_csv(filename + '.csv', index=None)
