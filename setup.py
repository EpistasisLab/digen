#!/usr/bin/env python

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='digen',
    version='0.0.2',
    author='Patryk Orzechowski',
    author_email=('patryk.orzechowski@gmail.com'),
    packages=['digen'],
    package_dir={'digen' : 'digen'},
    package_data={'digen': ['digen/data/*.pkl']},
    include_package_data=True,
    url='https://github.com/EpistasisLab/digen',
    description='DIGEN: Diverse Generative ML Benchmark',
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=True,
    install_requires=['pandas>=1.0.5',
                    'numpy>=1.19.5',
                    'scikit-learn>=0.22.2',
                    'optuna>=1.3.0',
                    'importlib-resources'
                   ],
    extras_require={
        'dev': ['numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas-profiling'],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
    ],
    keywords=['data mining', 'benchmark', 'machine learning', 'data analysis', 'datasets', 'data science']
)
