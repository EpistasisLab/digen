# Dataset SXGFL-KDR_0.111_0.819_769

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.8736  | 0.875181 |   0.79397  |            3 |            4 |         2 |
|  1 | XGBClassifier              | 0.8796  | 0.882838 |   0.777778 |            2 |            2 |         4 |
|  2 | LogisticRegression         | 0.5974  | 0.572443 |   0.58     |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.8649  | 0.842335 |   0.758621 |            5 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.66725 | 0.707311 |   0.60733  |            7 |            7 |         7 |
|  5 | SVC                        | 0.9327  | 0.913263 |   0.84466  |            1 |            1 |         1 |
|  6 | RandomForestClassifier     | 0.8678  | 0.875606 |   0.793814 |            4 |            3 |         2 |
|  7 | LGBMClassifier             | 0.865   | 0.868226 |   0.777778 |            5 |            5 |         4 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.17977492171788545, loss='deviance',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=53, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=13, presort='deprecated',
                           random_state=769, subsample=1.0, tol=1e-07,
                           validation_fraction=0.060000000000000005, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=2.241194140112394, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.3014681051405634, eval_metric='logloss',
              gamma=0.30000000000000004, gpu_id=-1, importance_type='gain',
              interaction_constraints=None, learning_rate=0.301468104,
              max_delta_step=0, max_depth=10, min_child_weight=1, missing=nan,
              monotone_constraints=None, n_estimators=55, n_jobs=0,
              num_parallel_tree=1, objective='binary:logistic',
              random_state=769, reg_alpha=2.24119425,
              reg_lambda=0.19174517314663223, scale_pos_weight=1, subsample=1,
              tree_method=None, validate_parameters=False, verbosity=None)
LogisticRegression(C=0.013186076929250699, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=769, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=47, p=3,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=9, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=769, splitter='best')
SVC(C=97.15664353756829, break_ties=False, cache_size=200, class_weight=None,
    coef0=4.6000000000000005, decision_function_shape='ovr', degree=2,
    gamma='auto', kernel='poly', max_iter=-1, probability=True,
    random_state=769, shrinking=True, tol=0.003943623922467287, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=9, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=43,
                       n_jobs=None, oob_score=False, random_state=769,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=9,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=99,
               n_jobs=-1, num_leaves=187, objective='binary', random_state=769,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='SXGFL-KDR_0.111_0.819_769-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='SXGFL-KDR_0.111_0.819_769-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='SXGFL-KDR_0.111_0.819_769-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='SXGFL-KDR_0.111_0.819_769-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='SXGFL-KDR_0.111_0.819_769-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='SXGFL-KDR_0.111_0.819_769-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/SXGFL-KDR_0.111_0.819_769.html)