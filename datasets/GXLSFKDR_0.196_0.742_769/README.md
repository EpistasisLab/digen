# Dataset GXLSFKDR_0.196_0.742_769

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9856  | 0.979659 |   0.964824 |            1 |            1 |         1 |
|  1 | XGBClassifier              | 0.9727  | 0.97634  |   0.923858 |            2 |            2 |         2 |
|  2 | LogisticRegression         | 0.4442  | 0.48704  |   0.470588 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.6331  | 0.629047 |   0.55914  |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.47255 | 0.514523 |   0.471795 |            7 |            7 |         7 |
|  5 | SVC                        | 0.78085 | 0.788189 |   0.690722 |            4 |            5 |         4 |
|  6 | RandomForestClassifier     | 0.7708  | 0.803444 |   0.690355 |            5 |            4 |         5 |
|  7 | LGBMClassifier             | 0.8796  | 0.883605 |   0.822335 |            3 |            3 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.8356146093834345, loss='deviance',
                           max_depth=7, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=77, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=18, presort='deprecated',
                           random_state=769, subsample=1.0, tol=1e-07,
                           validation_fraction=0.03, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=1.6361398357199706, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.13689026632897838, eval_metric='logloss', gamma=0.2,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.136890262, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=77, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=769,
              reg_alpha=1.63613987, reg_lambda=0.003471932368012841,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.0002973880938827089, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=769, solver='sag', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=14, p=5,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=769, splitter='best')
SVC(C=163.36808424238734, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=10.0, decision_function_shape='ovr',
    degree=3, gamma='scale', kernel='poly', max_iter=-1, probability=True,
    random_state=769, shrinking=True, tol=2.5554537223443177e-05,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=65,
                       n_jobs=None, oob_score=False, random_state=769,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=8,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=92,
               n_jobs=-1, num_leaves=75, objective='binary', random_state=769,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='GXLSFKDR_0.196_0.742_769-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='GXLSFKDR_0.196_0.742_769-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='GXLSFKDR_0.196_0.742_769-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='GXLSFKDR_0.196_0.742_769-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='GXLSFKDR_0.196_0.742_769-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='GXLSFKDR_0.196_0.742_769-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/GXLSFKDR_0.196_0.742_769.html)