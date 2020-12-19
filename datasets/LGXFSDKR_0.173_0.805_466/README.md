# Dataset LGXFSDKR_0.173_0.805_466

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9868  | 0.983375 |   0.974874 |            2 |            3 |         2 |
|  1 | XGBClassifier              | 0.9842  | 0.987439 |   0.948454 |            3 |            1 |         3 |
|  2 | LogisticRegression         | 0.4994  | 0.507763 |   0.455959 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.6597  | 0.669971 |   0.592593 |            7 |            7 |         7 |
|  4 | DecisionTreeClassifier     | 0.70355 | 0.737025 |   0.656566 |            6 |            6 |         6 |
|  5 | SVC                        | 0.7259  | 0.742662 |   0.680412 |            5 |            5 |         5 |
|  6 | RandomForestClassifier     | 0.8936  | 0.902459 |   0.78534  |            4 |            4 |         4 |
|  7 | LGBMClassifier             | 0.9894  | 0.987419 |   0.984925 |            1 |            1 |         1 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.09856925468972667, loss='deviance',
                           max_depth=8, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=18, presort='deprecated',
                           random_state=466, subsample=1.0, tol=1e-07,
                           validation_fraction=0.02, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.001309347764360703, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.09039419595826592, eval_metric='logloss', gamma=0.0,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.0903941989, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=92, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=466,
              reg_alpha=0.0013093478, reg_lambda=6.688827744595406e-05,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00019253501661343955, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=466, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=20, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=466, splitter='best')
SVC(C=0.016429988326301985, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=5.300000000000001,
    decision_function_shape='ovr', degree=5, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=466, shrinking=True,
    tol=0.009128945030583608, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=97,
                       n_jobs=None, oob_score=False, random_state=466,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=5,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=97,
               n_jobs=-1, num_leaves=103, objective='binary', random_state=466,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='LGXFSDKR_0.173_0.805_466-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='LGXFSDKR_0.173_0.805_466-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='LGXFSDKR_0.173_0.805_466-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='LGXFSDKR_0.173_0.805_466-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='LGXFSDKR_0.173_0.805_466-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='LGXFSDKR_0.173_0.805_466-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/LGXFSDKR_0.173_0.805_466.html)