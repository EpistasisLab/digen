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

GITHUB_URL = 'https://github.com/EpistasisLab/digen/blob/main/datasets'
suffix = '.tsv'


import os
from io import StringIO
import pkgutil



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from . import load_datasets

#from sklearn.metrics import roc_auc_score
#from sklearn.base import clone
#from sklearn.model_selection import train_test_split, StratifiedKFold
#from io import StringIO
#from sklearn.metrics import roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, f1_score


class Dataset:

    def __init__(self, dataset_name):
        df=pd.read_csv(StringIO(load_datasets()), sep=',', index_col='dataset')
        self.dataset_name=dataset_name
        self.model=df.loc[dataset_name]['indiv']
        self.hash=df.loc[dataset_name]['hash']


    def get_random_state(self, dataset_name):
        return int(dataset_name.split('_')[-1])

    def get_model(self):
        return self.model


    def get_hash(self):
        return self.hash

    def get_dataset_url(self, dataset_name, suffix=suffix):

        '''
        A method that downloads from DIGEN a dataset dataset_name from GITHUB_URL.
        '''

        if dataset_name:
            self.dataset_name=dataset_name
        dataset_url = '{GITHUB_URL}/{DATASET_NAME}/{DATASET_NAME}{SUFFIX}?raw=true'.format(
                                    GITHUB_URL=GITHUB_URL,
                                    DATASET_NAME=dataset_name,
                                    SUFFIX=suffix
                                    )

        re = requests.get(dataset_url)
        if re.status_code != 200:
            raise ValueError('Dataset not found in DIGEN.')
        print('Downloading '+dataset_name+' from '+ dataset_url)
        return dataset_url


    def load_dataset(self, separate_target=False, local_cache_dir=None):

        """
        Downloads a dataset from the DIGEN and returns it.

        Parameters
        ----------
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


        if local_cache_dir is None:
            local_cache_dir='.'
            if os.path.exists(os.path.join(local_cache_dir, self.dataset_name+suffix)):
                dataset_path = os.path.join(local_cache_dir, self.dataset_name+suffix)
            elif os.path.exists(os.path.join(local_cache_dir, self.dataset, self.dataset_name+suffix)):
                dataset_path = os.path.join(local_cache_dir, self.dataset, self.dataset_name+suffix)
            else:
                dataset_path = self.get_dataset_url(self.dataset_name, suffix)
        else:
            if os.path.exists(os.path.join(local_cache_dir, self.dataset_name+suffix)):
                dataset_path = os.path.join(local_cache_dir, self.dataset_name+suffix)
            else:
                raise OSError('File not found: '+os.path.join(local_cache_dir, self.dataset_name+suffix))
        dataset = pd.read_csv(dataset_path, sep='\t', compression='gzip')
        if not os.path.exists(os.path.join(local_cache_dir, self.dataset_name+suffix)):
            dataset.to_csv(os.path.join(local_cache_dir, self.dataset_name+suffix), sep='\t', compression='gzip', index=False)
        # prepare the output
        if separate_target:
            X = dataset.drop('target', axis=1).values
            y = dataset['target'].values
            return (X, y)
        else:
            return dataset






if __name__ == '__main__':
    dataset=Dataset('L-XG-FDSKR_0.188_0.875_7270')
    dataset.get_random_state('L-XG-FDSKR_0.188_0.875_7270')

