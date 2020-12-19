# Dataset GXLSFKDR_0.178_0.752_4426

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier |  0.9995 | 0.999522 |   0.989899 |            1 |            1 |         1 |
|  1 | XGBClassifier              |  0.9967 | 0.996625 |   0.959184 |            2 |            2 |         2 |
|  2 | LogisticRegression         |  0.4586 | 0.488095 |   0.445596 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       |  0.6441 | 0.641735 |   0.57754  |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     |  0.5764 | 0.559984 |   0.540816 |            7 |            7 |         7 |
|  5 | SVC                        |  0.7765 | 0.782018 |   0.680203 |            4 |            3 |         4 |
|  6 | RandomForestClassifier     |  0.7701 | 0.774813 |   0.670157 |            5 |            5 |         5 |
|  7 | LGBMClassifier             |  0.7939 | 0.779889 |   0.733668 |            3 |            4 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.8736678613486387, loss='exponential',
                           max_depth=9, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=78, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=20, presort='deprecated',
                           random_state=4426, subsample=1.0, tol=1e-07,
                           validation_fraction=0.04, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.011979552130222825, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.4085143668757905, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.40851438, max_delta_step=0, max_depth=7,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=4426,
              reg_alpha=0.011979552, reg_lambda=20.47086606523023,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.0008343143284614281, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=4426, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=1,
                     weights='uniform')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=4426, splitter='best')
SVC(C=347.3024943767419, break_ties=False, cache_size=200, class_weight=None,
    coef0=3.0, decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='poly', max_iter=-1, probability=True, random_state=4426,
    shrinking=True, tol=0.004366117368042038, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=99,
                       n_jobs=None, oob_score=False, random_state=4426,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=9,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=98,
               n_jobs=-1, num_leaves=22, objective='binary', random_state=4426,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='GXLSFKDR_0.178_0.752_4426-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='GXLSFKDR_0.178_0.752_4426-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='GXLSFKDR_0.178_0.752_4426-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='GXLSFKDR_0.178_0.752_4426-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='GXLSFKDR_0.178_0.752_4426-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='GXLSFKDR_0.178_0.752_4426-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/GXLSFKDR_0.178_0.752_4426.html)