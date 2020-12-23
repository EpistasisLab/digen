#!/usr/bin/env python

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='digen',
    version="0.0.1",
    author='Patryk Orzechowski',
    author_email=('patryk.orzechowski@gmail.com'),
    packages=find_packages(),
    package_data={'datasets': ['*.tsv']},
    include_package_data=True,
    url='https://github.com/EpistasisLab/digen',
    license='MIT',
    description='DIGEN: Diverse Generative Benchmark',
    zip_safe=True,
    install_requires=['pandas>=1.0.5',
                    'scikit-learn>=0.19.0',
                    'optuna>=1.3.0'
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
    ],
    keywords=['data mining', 'benchmark', 'machine learning', 'data analysis', 'datasets', 'data science'],
)
