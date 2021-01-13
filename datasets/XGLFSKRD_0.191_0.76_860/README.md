# Dataset XGLFSKRD_0.191_0.76_860

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier |  0.9739 | 0.979957 |   0.944162 |            2 |            2 |         2 |
|  1 | XGBClassifier              |  0.9874 | 0.98737  |   0.950495 |            1 |            1 |         1 |
|  2 | LogisticRegression         |  0.4976 | 0.508058 |   0.444444 |            7 |            7 |         7 |
|  3 | KNeighborsClassifier       |  0.6323 | 0.613002 |   0.635945 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     |  0.4675 | 0.505441 |   0.431818 |            8 |            8 |         8 |
|  5 | SVC                        |  0.8072 | 0.809608 |   0.72549  |            5 |            4 |         4 |
|  6 | RandomForestClassifier     |  0.8113 | 0.804783 |   0.722513 |            4 |            5 |         5 |
|  7 | LGBMClassifier             |  0.8997 | 0.894698 |   0.830918 |            3 |            3 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.5496309557301636, loss='deviance',
                           max_depth=8, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=66, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=17, presort='deprecated',
                           random_state=860, subsample=1.0, tol=1e-07,
                           validation_fraction=0.01, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=2.4276409387188376, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.11119826251359108, eval_metric='logloss', gamma=0.2,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.111198261, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=72, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=860,
              reg_alpha=2.42764091, reg_lambda=0.00021419808373663413,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=451.9212460720961, class_weight=None, dual=True,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=860, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=22, p=2,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=8, min_samples_split=9,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=860, splitter='best')
SVC(C=315.6058251991614, break_ties=False, cache_size=200, class_weight=None,
    coef0=9.5, decision_function_shape='ovr', degree=3, gamma='scale',
    kernel='poly', max_iter=-1, probability=True, random_state=860,
    shrinking=True, tol=0.0003291891725563437, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=8,
                       min_weight_fraction_leaf=0.0, n_estimators=73,
                       n_jobs=None, oob_score=False, random_state=860,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=7,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=91,
               n_jobs=-1, num_leaves=122, objective='binary', random_state=860,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='XGLFSKRD_0.191_0.76_860-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='XGLFSKRD_0.191_0.76_860-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='XGLFSKRD_0.191_0.76_860-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='XGLFSKRD_0.191_0.76_860-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='XGLFSKRD_0.191_0.76_860-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='XGLFSKRD_0.191_0.76_860-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/XGLFSKRD_0.191_0.76_860.html)