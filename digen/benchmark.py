# -*- coding: utf-8 -*-

"""
Copyright (c) 2020 Patryk Orzechowski | Epistasis Lab | University of Pennsylvania

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

GITHUB_URL = 'https://github.com/EpistasisLab/digen/tree/main/datasets'
suffix = '.tsv'


import os
#import pkgutil


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import pickle
import random


from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from io import StringIO
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, f1_score
from xgboost import XGBClassifier
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from . import initialize
from .dataset import Dataset



class Benchmark:
    '''
    The main class of DIGEN.

    This class implements base methods for benchmarking methods.
    In order to keep it consistent with the benchmark, the defaults shouldn't be modified.

    Parameters:
    ----------

    n_trials : int, default=200
        The maximal number of combinations of hyper-parameters optimizations performed during benchmarking ML method.
        For further reference, see Optuna.

    timeout : int, default=10000
        The maximal time that is allowed for the ML method to run all the optimizations..
        After timeout is reached, the current experiments are completed and no further optimizations can be started.

    n_split : int, default=10
        Number of splits for k-fold cross validation

    '''

    def __init__(self, n_trials=200, timeout=10000, n_splits=10):

        data = initialize()
        df=pd.DataFrame(data)
        assert(len(df) > 0)
        df['fpr'] = df['fpr'].str.slice(1,-1).str.split(', ').apply(lambda x : np.array([float(i) for i in x]))
        df['tpr'] = df['tpr'].str.slice(1,-1).str.split(', ').apply(lambda x : np.array([float(i) for i in x]))
        df['prec'] = df['prec'].str.slice(1,-1).str.split(', ').apply(lambda x : np.array([float(i) for i in x]))
        df['rec'] = df['rec'].str.slice(1,-1).str.split(', ').apply(lambda x : np.array([float(i) for i in x]))
        self.data = df
        self.dataset_names = pd.unique(df['dataset']).tolist()
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_splits = n_splits



    def list_datasets(self):
        '''
        This method lists all datasets

        Returns
        --------
        dataset_names : a list of strings
            List of all names of the datasets included in DIGEN.
        '''
        return self.dataset_names



    def load_dataset(self, dataset_name, separate_target=False, local_cache_dir=None):

        """Downloads a dataset from the DIGEN and returns it. For convenience, instead of using Dataset interface.

        Parameters
        ----------
        dataset_name : str
            The name of the data set to load from DIGEN.
        separate_target : bool (default: False)
            Should the target variable be kept within the array in scikit-learn format, or the features separate as NumPy arrays.
        local_cache_dir: str (default: None)
            The directory on your local machine to store the data files.
            If None, then the local data cache will not be used and the datasets downloaded from Github.

        Returns
        ----------
        dataset: pd.DataFrame or (array-like, array-like)
            if separate_target == False: A pandas DataFrame containing the fetched data set.
            if separate_target == True: A tuple of NumPy arrays containing (features, labels)

        """

        dataset = Dataset(dataset_name)
        return dataset.load_dataset(separate_target=separate_target, local_cache_dir=local_cache_dir)



    def optimize(self, est, parameter_scopes, datasets=None, storage='sqlite:///default.db', local_cache_dir=None):
        '''
        The method that optimizes hyper-parameters for a single or multiple DIGEN datasets.

        Parameters
        ----------

        est : sklearn.base.BaseEstimator
            A method that will be optimized and benchmarked against DIGEN.

        parameter_scopes : dict
            A dictionary containing hyper parameters of the benchmarked ML method as well as their distributions.
            Refer to Optuna Trial:
            https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args
            https://optuna.readthedocs.io/en/v1.4.0/reference/trial.html

        datasets : a string or a list
            The name(s) of the dataset(s) that DIGEN will be run on.

        storage : string, default: local file named default.db
            The link to SQLite dataset which will store the optimizations from Optuna.
        '''

        best_models = dict()
        if (datasets == None):
            datasets = self.list_datasets()
        if not isinstance(datasets, list):
            datasets = [datasets]

        for dataset_name in datasets:
            print('Optimizing ' + est.__class__.__name__ + ' on ' + dataset_name)

            dataset = Dataset(dataset_name)
            random_state = dataset.get_random_state(dataset_name)
            random.seed(random_state)
            np.random.seed(random_state)

            self.random_state=random.getstate()
            self.random_state_np=np.random.get_state()

            sampler = optuna.samplers.TPESampler(seed=random_state)  # Make the sampler behave in a deterministic way.
            study = optuna.create_study(study_name=dataset_name + '-' +  est.__class__.__name__ ,
                                  direction='maximize',
                                  sampler=sampler,
                                  storage=storage, load_if_exists=True)


#            X, y = dataset.load_dataset(dataset_name, separate_target=True, local_cache_dir=local_cache_dir)
            X, y = dataset.load_dataset(separate_target=True, local_cache_dir=local_cache_dir)

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

            study.optimize(lambda trial: self._objective(trial, X_train, y_train, est, parameter_scopes, random_state), n_trials=self.n_trials, timeout=self.timeout)
            best_models[dataset_name] = clone(est).set_params(**study.best_trial.user_attrs['params'])
#            best_models[dataset_name] = clone(est.__class__(**study.best_trial.user_attrs['params']))
#            print(str(best_models))
        return best_models




    def evaluate(self, est, dataset_name, local_cache_dir=None):
        '''
        A method that calculates different performance metrics for the ML method with parameters.
        This function doesn't tune the parameters.

        Parameters
        ----------

        est : sklearn.base.BaseEstimator
            A method that will be timized and benchmarked against DIGEN.

        '''
        dataset = Dataset(dataset_name)
        random_state = dataset.get_random_state(dataset_name)


        X, y = dataset.load_dataset(separate_target=True, local_cache_dir=local_cache_dir)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)	

        est.fit(X_train, y_train)
        yproba = est.predict_proba(X_test)[::,1]
        y_pred = est.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test,  yproba)
        auroc = roc_auc_score(y_test, yproba)
        #average_precision_score(y_test,yproba)
        prec,rec, _ = precision_recall_curve(y_test, yproba)
        return {
                    'dataset' : dataset_name,
                    'classifier' : est.__class__.__name__,
                    'fpr' : fpr,
                    'tpr' : tpr,
                    'prec' : prec,
                    'rec' : rec,
                    'auroc' : auroc,
                    'f1_score' : f1_score(y_test,y_pred),
                    'auprc' : auc(rec,prec)
                }




    def _objective(self, trial, X, y, estimator, parameter_scopes, random_state):
        '''
        An internal method that sets Optuna parameters and objective for hyper-parameter optimization
        '''
        est = clone(estimator).set_params(**parameter_scopes(trial))
#        est = clone(estimator.__class__(**parameter_scopes(trial)))

        for a in ['random_state','seed']:
            if hasattr(est,a):
                setattr(est,a,random_state)

        splits_auc=[]
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in cv.split(X, y):
            split_num = len(splits_auc)

            if isinstance(X, np.ndarray):
                X_train = X[train_idx]
                X_test = X[test_idx]
            else:
                X_train = X.iloc[train_idx, :]
                X_test = X.iloc[test_idx, :]
            if isinstance(y, np.ndarray):
                y_train = y[train_idx]
                y_test = y[test_idx]
            else:
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

            est.fit(X_train,y_train)
            auroc_test = roc_auc_score(y_test, est.predict(X_test))
            trial.set_user_attr('split_id', split_num)
            trial.set_user_attr('fold' + str(split_num) + '_auroc', auroc_test)
            splits_auc.append(auroc_test)

        trial.set_user_attr('estimator', str(estimator.__class__.__name__))
        trial.set_user_attr('auroc', np.mean(splits_auc))
        trial.set_user_attr('seed', random_state)
        trial.set_user_attr('params', est.get_params())
        return np.mean(splits_auc)




    def plot_roc(self, dataset_name, ax=None, new_results=None, **kwargs):
        '''
        A method that plots an ROC curve chart for a given dataset with methods included in DIGEN with or without the additional benchmarked method.

        Parameters
        ----------
        dataset_name : str
            The name of the data set to load from DIGEN.
        new_result : dict
            The result of evaluation of the given estimator on the dataset_name 
            For further reference, see: evaluate 

        '''

        df=self.data[self.data['dataset'] == dataset_name]
        df.reset_index(inplace=True)
        assert(len(df)>0)
        linestyles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
        colors = ['0.4','0.4','0.4','0.4','0.8','0.8','0.8','0.8']



        if ax is None:
            fig, ax = plt.subplots(figsize=(16,12))

        if new_results is not None:
            assert(isinstance(new_results, dict))
            df=df.append(new_results, ignore_index=True)
            linestyles.append('-')
            colors.append('red')
        linestyles = iter(linestyles)
        colors = iter(colors)
        for i in df.index:
            ax.plot(df.loc[i]['fpr'],
                df.loc[i]['tpr'],
                color = next(colors), linestyle=next(linestyles),
                label = "{}, AUC={:.3f}".format(df.loc[i]['classifier'], df.loc[i]['auroc']), **kwargs)

        ax.plot([0,1], [0,1], color='black', linestyle='--')
        ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
        ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel("False Positive Rate", fontsize=24)
        ax.set_ylabel("True Positive Rate", fontsize=24)
        ax.set_title('ROC Curves Comparison', fontweight='bold', fontsize=28)
        ax.legend(prop={'size':13}, loc='lower right')
        plt.gcf().subplots_adjust(bottom=0.15)
        return fig, ax



    def plot_prc(self, dataset_name, ax=None, new_results=None, **kwargs):

        '''
        A method that plots an PRC curve chart for a given dataset with methods included in DIGEN with or without the additional benchmarked method.

        Parameters
        ----------
        dataset_name : str
            The name of the data set to load from DIGEN.
        new_result : dict
            The result of evaluation of the given estimator on the dataset_name 
            For further reference, see: evaluate 
        '''



        df=self.data[self.data['dataset'] == dataset_name]

        df.reset_index(inplace=True)
        assert(len(df)>0)
        fig = plt.figure(figsize=(16,12))
        linestyles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
        colors = ['0.4','0.4','0.4','0.4','0.8','0.8','0.8','0.8']

        if new_results is not None:
            assert(isinstance(new_results,dict))
            df=df.append(new_results, ignore_index=True)
            linestyles.append('-')
            colors.append('red')

        if ax is None:
            fig, ax = plt.subplots(figsize=(16,12))
        linestyles = iter(linestyles)
        colors = iter(colors)

        for i in df.index:
            ax.plot(df.loc[i]['rec'],
                df.loc[i]['prec'],
                color = next(colors), linestyle=next(linestyles),
                label = "{}, f1_score={:.3f}, auprc={:.3f}".format(df.loc[i]['classifier'],df.loc[i]['f1_score'], df.loc[i]['auprc']), **kwargs)

        ax.plot([0,1], [0.5,0.5], color='black', linestyle='--')
        ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
        ax.set_xlabel("Recall", fontsize=24)
        ax.set_yticks(np.arange(0.4, 1.1, step=0.1))
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylabel("Precision", fontsize=24)
        ax.set_title('Precision-Recall Curve Comparison', fontweight='bold', fontsize=28)
        ax.legend(prop={'size':13}, loc='lower right')
        plt.gcf().subplots_adjust(bottom=0.15)
    
        return fig, ax


    def plot_parallel_coordinates(self, datasets=None, ax=None, benchmarked_results=None, **kwargs):

        '''
        A method that plots a parallel coordinate plot for the whole benchmark with or without the additional benchmarked method.
        If no additional method is presented, the results are averaged to a mean.
        performance of all of the methods is averag

        Parameters
        ----------
        datasets : list(str)
            The name of the data set to load from DIGEN.
        new_result : dict
            The result of evaluation of the given estimator on the dataset_name 
            For further reference, see: evaluate 
        '''

        if ax is None:
            fig, ax = plt.subplots(figsize=(25,12))


        if datasets is None:
            datasets=self.list_datasets()
        if not isinstance(datasets, list):
            datasets = [datasets]

        # getting performance of all the classifiers, and limiting to the listed datasets
        df=self.data.pivot('classifier', columns='dataset', values='auroc')[datasets]

        # if no dataset is added, just subtract mean performance for all datasets
        if benchmarked_results is None:
            df=df-df.mean()
        #otherwise, use  provided results as a reference
        else:
            df=df-pd.Series(benchmarked_results)

        #columns - classifiers, rows-datasets
        df=df.transpose()
        df['legend']=df.index
        ax=pd.plotting.parallel_coordinates(df, 'legend', #color_continuous_scale=px.colors.sequential.Blues,
                                     #color=["lime", "tomato","dodgerblue"],
                                     alpha=0.2)

        if benchmarked_results is None:
            plt.title("Performance of classifiers compared to mean AUROC on DIGEN benchmark")
            plt.ylabel('Difference vs mean AUROC value')
#            plt.title("Performance of classifiers compared to mean AUROC on DIGEN benchmark", fontsize=28)
#            plt.ylabel('Difference vs mean AUROC value', fontsize=24)

        else:
            plt.title("Performance of classifiers compared vs the method on DIGEN benchmark", fontsize=28)
            plt.ylabel('Difference in AUROC vs benchmark performance', fontsize=24)
        plt.plot([0,7], [0,0], color='red', linestyle='--')
        ax.get_legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)
        plt.xlabel('Classifiers')
        plt.xticks(rotation=90)
#        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#        ax.set_facecolor("white")
#        plt.gcf().subplots_adjust(bottom=0.15)

        return fig, ax



    def plot_heatmap(self, datasets=None, ax=None, benchmarked_results=None, **kwargs):

        '''
        A method that plots a parallel coordinate plot for the whole benchmark with or without the additional benchmarked method.
        If no additional method is presented, the results are averaged to a mean.
        performance of all of the methods is averag

        Parameters
        ----------
        dataset_list : list(str)
            The name of the data set to load from DIGEN.
        new_result : dict
            The result of evaluation of the given estimator on the dataset_name 
            For further reference, see: evaluate 
        '''

#        if ax is None:
#            fig, ax = plt.subplots()


        if datasets is None:
            datasets=self.list_datasets()
        if not isinstance(datasets, list):
            datasets = [datasets]
        # getting performance of all the classifiers, and limiting to the listed datasets
        df=self.data.pivot('classifier', columns='dataset', values='auroc')[datasets]
        if benchmarked_results is not None:
            df=df.append(pd.Series(benchmarked_results))
        df=df.transpose()
        g=sns.clustermap(df, cmap='Blues',  yticklabels=True, row_cluster=False)

#        plt.title("Similiarity of AUROCs scores of ML classifiers across different DIGEN problems") #fontsize=28
#        plt.xticks(fontsize=20, rotation=90)
#        plt.xticks(rotation=90)

#
#
#        plt.ylabel('Area under ROC curve') #fontsize=24
##        col=ax.cax.get_position()
#        ax.cax.set_position([col.x0+1, col.y0-0.35, col.width, col.height])
#        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#        ax.set_facecolor("white")
#        plt.gcf().subplots_adjust(bottom=0.15)


        g.fig.set_figheight(16)
        g.fig.set_figwidth(12)
        col=g.cax.get_position()
        g.cax.set_position([col.x0+1, col.y0-0.35, col.width, col.height])

        rcParams.update({'figure.autolayout': True})
        return g, ax






'''
est = ExtraTreesClassifier()


best_models = digen.optimize_all(est, dataset_name)
for best_model in best_models:
    digen.plot_auroc(best_model, dataset_name)
    digen.plot_auprc(best_model, dataset_name)
digen.plot_summary(best_models)
digen.plot_ranking(best_models)
figen.plot_complexity(best_models)

        ##############TODO :SET HYPERPARAMETERS
        parameter_scopes=hyperparams_ExtraTrees(trial)
digen.plot_complexity()
digen.plot

'''




if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.list_datasets()
#    dataset_name = 'F-GLXDKRS_0.175_0.861_5191'
#    best_model = benchmark.optimize(est, dataset_name)
#    benchmark.plot_auroc(best_model, dataset_name)
#    benchmark.plot_auprc(best_model, dataset_name)

#    est = XGBClassifier()
#    optimize(est,'L-XG-FDSKR_0.188_0.875_7270')


