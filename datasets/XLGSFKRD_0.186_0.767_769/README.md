# Dataset XLGSFKRD_0.186_0.767_769

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9637  | 0.968363 |   0.896907 |            3 |            2 |         2 |
|  1 | XGBClassifier              | 0.9812  | 0.974233 |   0.935323 |            1 |            1 |         1 |
|  2 | LogisticRegression         | 0.5156  | 0.549146 |   0.502463 |            7 |            7 |         7 |
|  3 | KNeighborsClassifier       | 0.6447  | 0.612847 |   0.586387 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.50095 | 0.525893 |   0.49505  |            8 |            8 |         8 |
|  5 | SVC                        | 0.8128  | 0.829985 |   0.723005 |            4 |            4 |         4 |
|  6 | RandomForestClassifier     | 0.7497  | 0.749851 |   0.714286 |            5 |            5 |         5 |
|  7 | LGBMClassifier             | 0.9672  | 0.965365 |   0.896907 |            2 |            3 |         2 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.6708576662379794, loss='exponential',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=62, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=19, presort='deprecated',
                           random_state=769, subsample=1.0, tol=1e-07,
                           validation_fraction=0.06999999999999999, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=1.8461482636409923, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.30411045099788636, eval_metric='logloss', gamma=0.4,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.304110438, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=80, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=769,
              reg_alpha=1.84614825, reg_lambda=3.74902953307975,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00027402341452476356, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=769, solver='sag', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=14, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=8,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=769, splitter='best')
SVC(C=21246.54128210218, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=4.7, decision_function_shape='ovr', degree=3,
    gamma='scale', kernel='poly', max_iter=-1, probability=True,
    random_state=769, shrinking=True, tol=0.0023479380992256263, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=9, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=58,
                       n_jobs=None, oob_score=False, random_state=769,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=10,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=96,
               n_jobs=-1, num_leaves=215, objective='binary', random_state=769,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='XLGSFKRD_0.186_0.767_769-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='XLGSFKRD_0.186_0.767_769-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='XLGSFKRD_0.186_0.767_769-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='XLGSFKRD_0.186_0.767_769-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='XLGSFKRD_0.186_0.767_769-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='XLGSFKRD_0.186_0.767_769-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/XLGSFKRD_0.186_0.767_769.html)