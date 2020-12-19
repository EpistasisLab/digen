# Dataset FLGXDKSR_0.157_0.871_6265

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.98685 | 0.982778 |   0.97561  |            3 |            3 |         1 |
|  1 | XGBClassifier              | 0.9845  | 0.981523 |   0.970297 |            4 |            4 |         2 |
|  2 | LogisticRegression         | 0.5742  | 0.570902 |   0.590674 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.7543  | 0.768902 |   0.677419 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.9731  | 0.975582 |   0.970297 |            5 |            5 |         2 |
|  5 | SVC                        | 0.7017  | 0.689316 |   0.676617 |            7 |            7 |         6 |
|  6 | RandomForestClassifier     | 0.99735 | 0.997311 |   0.970297 |            1 |            1 |         2 |
|  7 | LGBMClassifier             | 0.9935  | 0.993774 |   0.94686  |            2 |            2 |         5 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.047471047474927947, loss='deviance',
                           max_depth=5, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=4, presort='deprecated',
                           random_state=6265, subsample=1.0, tol=1e-07,
                           validation_fraction=0.02, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.7976085817221445, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=1.189053209220623e-07, eval_metric='logloss', gamma=0.2,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=1.18905319e-07, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=60, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=6265,
              reg_alpha=0.797608554, reg_lambda=6.923128813527922,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.020450652968583574, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=6265, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=16, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=6265, splitter='best')
SVC(C=19.087433268954488, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=3.0, decision_function_shape='ovr', degree=2,
    gamma='scale', kernel='rbf', max_iter=-1, probability=True,
    random_state=6265, shrinking=True, tol=4.516694353534716e-05,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=6, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=82,
                       n_jobs=None, oob_score=False, random_state=6265,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=10,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=78,
               n_jobs=-1, num_leaves=12, objective='binary', random_state=6265,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (200 experiments per ML method)</summary>
<img src='FLGXDKSR_0.157_0.871_6265-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='FLGXDKSR_0.157_0.871_6265-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='FLGXDKSR_0.157_0.871_6265-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='FLGXDKSR_0.157_0.871_6265-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='FLGXDKSR_0.157_0.871_6265-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='FLGXDKSR_0.157_0.871_6265-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://github.io/athril/digen-test/docs/profile/FLGXDKSR_0.157_0.871_6265.html)