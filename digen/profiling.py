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



import pandas as pd
from pandas_profiling import ProfileReport
import pathlib

#from digen import Benchmark
from .benchmark import (
    Benchmark
)

from .dataset import (
    Dataset
)



def profile_dataset(dataname, write_dir, repository='../datasets/'):
    '''
    Performs pandas profiling of datasets and writes the results to write_dir
    '''

    b=Benchmark()
    print(f'Processing {dataname}')
    df = b.load_dataset(dataname, local_cache_dir=repository)
    write_path = write_dir.joinpath(dataname + '.html')
    profile = ProfileReport(df, title=dataname, explorative=True)
    profile.to_file(write_path)


if __name__ =='__main__':
    write_dir = pathlib.Path('docs/profile/')
    write_dir.mkdir(exist_ok=True)

    b=Benchmark()
    datasets = b.list_datasets()

    for dataset in datasets:
        write_path = write_dir.joinpath(dataset + '.html')
        profile_dataset(dataset, write_dir)
