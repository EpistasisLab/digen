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
import pkgutil


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from io import StringIO
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, f1_score


class Dataset:

    def __init__(self, dataset_name):
        self.dataset_name=dataset_name

    def get_random_state(self, dataset_name):
        return int(dataset_name.split('_')[-1])


if __name__ == '__main__':
    dataset=Dataset('L-XG-FDSKR_0.188_0.875_7270')
    dataset.get_random_state('L-XG-FDSKR_0.188_0.875_7270')


