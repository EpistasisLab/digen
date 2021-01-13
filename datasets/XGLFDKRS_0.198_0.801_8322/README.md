# Dataset XGLFDKRS_0.198_0.801_8322

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9982  | 0.998468 |   0.994975 |            2 |            2 |         1 |
|  1 | XGBClassifier              | 0.9998  | 0.999803 |   0.99     |            1 |            1 |         2 |
|  2 | LogisticRegression         | 0.5292  | 0.508041 |   0.527363 |            7 |            7 |         7 |
|  3 | KNeighborsClassifier       | 0.7051  | 0.691421 |   0.646465 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.80605 | 0.808443 |   0.753623 |            5 |            5 |         5 |
|  5 | SVC                        | 0.4771  | 0.458188 |   0.505155 |            8 |            8 |         8 |
|  6 | RandomForestClassifier     | 0.9009  | 0.881566 |   0.832536 |            4 |            4 |         4 |
|  7 | LGBMClassifier             | 0.9949  | 0.995259 |   0.95098  |            3 |            3 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.04887848178761428,
                           loss='exponential', max_depth=9, max_features=None,
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=100, n_iter_no_change=20,
                           presort='deprecated', random_state=8322,
                           subsample=1.0, tol=1e-07, validation_fraction=0.02,
                           verbose=0, warm_start=False)
XGBClassifier(alpha=2.6589247688993733, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.09384908462356675, eval_metric='logloss', gamma=0.0,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.0938490853, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=44, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=8322,
              reg_alpha=2.65892482, reg_lambda=0.00015367086121316636,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00010226147853200833, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=8322, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=16, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=9, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=12, min_samples_split=18,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=8322, splitter='best')
SVC(C=79611.69522329033, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=4.0, decision_function_shape='ovr', degree=2,
    gamma='auto', kernel='poly', max_iter=-1, probability=True,
    random_state=8322, shrinking=True, tol=0.0007548761162901196,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=77,
                       n_jobs=None, oob_score=False, random_state=8322,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=10,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=88,
               n_jobs=-1, num_leaves=165, objective='binary', random_state=8322,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='XGLFDKRS_0.198_0.801_8322-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='XGLFDKRS_0.198_0.801_8322-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='XGLFDKRS_0.198_0.801_8322-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='XGLFDKRS_0.198_0.801_8322-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='XGLFDKRS_0.198_0.801_8322-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='XGLFDKRS_0.198_0.801_8322-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/XGLFDKRS_0.198_0.801_8322.html)