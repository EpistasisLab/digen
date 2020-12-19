# Dataset RXSLKGFD_0.03_0.965_769

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9732  | 0.979681 |   0.923858 |            6 |            4 |         5 |
|  1 | XGBClassifier              | 0.9818  | 0.987784 |   0.948454 |            2 |            2 |         3 |
|  2 | LogisticRegression         | 0.9828  | 0.98903  |   0.979592 |            1 |            1 |         1 |
|  3 | KNeighborsClassifier       | 0.9743  | 0.978975 |   0.904523 |            5 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.8886  | 0.906842 |   0.797872 |            8 |            8 |         8 |
|  5 | SVC                        | 0.9812  | 0.987679 |   0.96     |            3 |            2 |         2 |
|  6 | RandomForestClassifier     | 0.9605  | 0.965001 |   0.899471 |            7 |            7 |         7 |
|  7 | LGBMClassifier             | 0.97465 | 0.979708 |   0.932642 |            4 |            4 |         4 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.36293905524267567, loss='deviance',
                           max_depth=1, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=10, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=12, presort='deprecated',
                           random_state=769, subsample=1.0, tol=1e-07,
                           validation_fraction=0.18000000000000002, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.016720154211910067, base_score=0.5, booster='gblinear',
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, eta=0.00047632499109246344,
              eval_metric='logloss', gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.000476324989, max_delta_step=None, max_depth=2,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=41, n_jobs=0, num_parallel_tree=None,
              objective='binary:logistic', random_state=769,
              reg_alpha=0.0167201534, reg_lambda=0.0005934186115412309,
              scale_pos_weight=1, subsample=None, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.13076773279674006, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=769, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=99, p=5,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=8, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=13,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=769, splitter='best')
SVC(C=147.24592418153992, break_ties=False, cache_size=200, class_weight=None,
    coef0=3.3000000000000003, decision_function_shape='ovr', degree=4,
    gamma='auto', kernel='linear', max_iter=-1, probability=True,
    random_state=769, shrinking=True, tol=0.00396017637772865, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, n_estimators=98,
                       n_jobs=None, oob_score=False, random_state=769,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=10,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=98,
               n_jobs=-1, num_leaves=3, objective='binary', random_state=769,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='RXSLKGFD_0.03_0.965_769-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='RXSLKGFD_0.03_0.965_769-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='RXSLKGFD_0.03_0.965_769-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='RXSLKGFD_0.03_0.965_769-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='RXSLKGFD_0.03_0.965_769-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='RXSLKGFD_0.03_0.965_769-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/RXSLKGFD_0.03_0.965_769.html)