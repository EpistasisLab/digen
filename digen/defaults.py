# -*- coding: utf-8 -*-

"""
Copyright (c) 2021 Patryk Orzechowski | Epistasis Lab | University of Pennsylvania

DIGEN was developed at the University of Pennsylvania by Patryk Orzechowski (patryk.orzechowski@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



'''
This file contains hyper-parameters of ML methods used during evolutionary search
'''



def params_LogisticRegression(trial):
    params={}
    params['solver'] = trial.suggest_categorical(name='solver',choices=['newton-cg', 'lbfgs', 'liblinear','sag','saga'])
    params['dual']= False
    params['penalty'] = 'l2'
    params['C'] = trial.suggest_loguniform('C', 1e-4, 1e4)
    params['l1_ratio'] = None
    if params['solver'] == 'liblinear':
        params['penalty'] = trial.suggest_categorical(name='penalty',choices=['l1','l2'])
        if params['penalty'] == 'l2':
            params['dual'] = trial.suggest_categorical(name='dual',choices=[True, False])
        else:
            params['penalty'] = 'l1'

    params['class_weight'] = trial.suggest_categorical(name='class_weight',choices=['balanced'])
    param_grid= {'solver' : params['solver'],
                 'penalty' : params['penalty'],
                 'dual' : params['dual'],
                 'multi_class' : 'auto',
                 'l1_ratio' : params['l1_ratio'], 
                 'C' : params['C'],
    }
    return param_grid


def params_KNeighborsClassifier(trial):
    return {
        'n_neighbors' : trial.suggest_int('n_neighbors', 1,100),
        'weights' : trial.suggest_categorical('weights',['uniform', 'distance']),
        'p' : trial.suggest_int('p',1,5),
        'metric' : trial.suggest_categorical('metric', ['euclidean', 'minkowski'])
    }

def params_DecisionTreeClassifier(trial):
    return  {
        'criterion' : trial.suggest_categorical('criterion',['gini', 'entropy']),
        'max_depth' : trial.suggest_int('max_depth', 1, 10),
        #'max_depth_factor' : trial.suggest_discrete_uniform('max_depth_factor', 0, 2,0.1),
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf',1, 20),
        'min_weight_fraction_leaf' : 0.0,
        'max_features' : trial.suggest_categorical('max_features',[None, 'auto', 'log2']),
        'max_leaf_nodes' : None, 
    }
  

def params_SVC(trial):
    return {
        'kernel': trial.suggest_categorical(name='kernel',choices=['poly', 'rbf','linear', 'sigmoid']),
        'C' : trial.suggest_loguniform('C', 1e-2, 1e5), 
        'gamma' : trial.suggest_categorical(name='gamma',choices=['scale', 'auto']),
        'degree' : trial.suggest_int('degree',2, 5),
        'class_weight' : trial.suggest_categorical(name='class_weight',choices=[None, 'balanced']),
        'coef0' : trial.suggest_discrete_uniform('coef0', 0, 10, 0.1),
        'tol' : trial.suggest_loguniform('tol', 1e-5, 1e-2),
        'probability' : trial.suggest_categorical(name='probability',choices=[True]),
    }

def params_RandomForestClassifier(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators',10,100),
        'criterion' : trial.suggest_categorical(name='criterion',choices=['gini', 'entropy']), 
        'max_depth' : trial.suggest_int('max_depth', 1, 10),
        'max_features' : trial.suggest_categorical('max_features',[None, 'auto','log2']),
        'bootstrap' : True,
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 20),
    }
    return params




def params_GradientBoostingClassifier(trial):
    params = {
        'loss' : trial.suggest_categorical(name='loss',choices=['deviance', 'exponential']),
       'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-2, 1),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 200),
        'max_depth' : trial.suggest_int('max_depth', 1, 10),
        'max_leaf_nodes' : None, 
        'tol' : 1e-7,
        'n_iter_no_change' : trial.suggest_int('n_iter_no_change', 1, 20),
        'validation_fraction' : trial.suggest_discrete_uniform('validation_fraction', 0.01, 0.31, 0.01)
    }
    return params

def params_XGBClassifier(trial):
    return {
            'booster' : trial.suggest_categorical(name='booster',choices=['gbtree', 'dart']),
            'n_estimators' : trial.suggest_int('n_estimators',10,100),
            'objective' : 'binary:logistic',
            'reg_lambda' : trial.suggest_loguniform('reg_lambda', 1e-5, 1e2),
            'alpha' : trial.suggest_loguniform('alpha',1e-5, 1e2),
            'gamma' : trial.suggest_discrete_uniform('gamma', 0, 0.5, 0.1),
            'eta' : trial.suggest_loguniform('eta', 1e-8, 1),
            'max_depth' : trial.suggest_int('max_depth',1,10),
            'eval_metric' : 'logloss',
            'tree_method' : 'exact',
            'nthread' : 1,
            'use_label_encoder' : False,
        }




def params_LGBMClassifier(trial):
    params= {
        'objective' : 'binary',
        'metric' : 'binary_logloss',
        'boosting_type' : trial.suggest_categorical(name='boosting_type',choices=['gbdt', 'dart', 'goss']),
        'num_leaves' : trial.suggest_int('num_leaves', 2,256),
        'max_depth' : trial.suggest_int('max_depth', 1, 10 ),
        'n_estimators' : trial.suggest_int('n_estimators',10,100), #200-6000 by 200
        'deterministic' : True,
        'force_row_wise' : True,
        'njobs' : 1,
    }
    if 2**params['max_depth']>params['num_leaves']:
        params['num_leaves']=2**params['max_depth']
    return params



def set_hyper_params(trial,est, n_features=1):
    if est == 'XGBClassifier':
        return params_XGBClassifier(trial)
    elif est == 'SVC':
        return params_SVC(trial)
    elif est == 'DecisionTreeClassifier':
        return params_DecisionTreeClassifier(trial)
    elif est == 'RandomForestClassifier':
        return params_RandomForestClassifier(trial)
    elif est == 'LGBMClassifier':
        return params_LGBMClassifier(trial)
    elif est == 'GradientBoostingClassifier':
        return params_GradientBoostingClassifier(trial)
    elif est == 'KNeighborsClassifier':
        return params_KNeighborsClassifier(trial)
    elif est == 'LogisticRegression':
        return params_LogisticRegression(trial)
    return None



def reclassify(y, threshold):
    ''' 
    Calculates the endpoint based on the reus
    
    Parameters
    ----------
    y : float 
        Original continuous value calculated using a given GP tree
    threshold : float
        Percentage of 1's in the reclassified endpoint (threshold is within (0,1))

    Output
    ------

    new_y : int
        binary endpoint (outcome)
    '''
    
    order=np.argsort(y)
    new_y=np.zeros(len(y))
    new_y[order[int(len(y)*threshold): ]]=1
    return new_y
    
    
