# Dataset SXGLKFDR_0.163_0.69_5578

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.7924  | 0.796957 |   0.683417 |            3 |            3 |         4 |
|  1 | XGBClassifier              | 0.83    | 0.832549 |   0.755981 |            2 |            2 |         2 |
|  2 | LogisticRegression         | 0.4121  | 0.436353 |   0.507042 |            8 |            8 |         7 |
|  3 | KNeighborsClassifier       | 0.7002  | 0.707902 |   0.625    |            5 |            4 |         6 |
|  4 | DecisionTreeClassifier     | 0.45255 | 0.474972 |   0.333333 |            7 |            7 |         8 |
|  5 | SVC                        | 0.9001  | 0.907686 |   0.808081 |            1 |            1 |         1 |
|  6 | RandomForestClassifier     | 0.6721  | 0.675229 |   0.635514 |            6 |            6 |         5 |
|  7 | LGBMClassifier             | 0.7621  | 0.689347 |   0.721154 |            4 |            5 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.3374370511235407, loss='exponential',
                           max_depth=8, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=10, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=20, presort='deprecated',
                           random_state=5578, subsample=1.0, tol=1e-07,
                           validation_fraction=0.01, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=1.3187934485360546e-05, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.7127550854382081, eval_metric='logloss', gamma=0.0,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.712755084, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=93, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=5578,
              reg_alpha=1.31879342e-05, reg_lambda=9.041757120219527,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.0007069332604796826, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5578, solver='sag', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=24, p=3,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=7, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=5578, splitter='best')
SVC(C=253.28796694818027, break_ties=False, cache_size=200, class_weight=None,
    coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='poly', max_iter=-1, probability=True, random_state=5578,
    shrinking=True, tol=0.0004415410179212527, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=58,
                       n_jobs=None, oob_score=False, random_state=5578,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=8,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
               n_jobs=-1, num_leaves=55, objective='binary', random_state=5578,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='SXGLKFDR_0.163_0.69_5578-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='SXGLKFDR_0.163_0.69_5578-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='SXGLKFDR_0.163_0.69_5578-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='SXGLKFDR_0.163_0.69_5578-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='SXGLKFDR_0.163_0.69_5578-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='SXGLKFDR_0.163_0.69_5578-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/SXGLKFDR_0.163_0.69_5578.html)