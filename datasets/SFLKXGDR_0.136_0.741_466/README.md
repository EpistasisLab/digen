# Dataset SFLKXGDR_0.136_0.741_466

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.7628  | 0.77712  |   0.698565 |            6 |            6 |         4 |
|  1 | XGBClassifier              | 0.7813  | 0.801365 |   0.68932  |            5 |            4 |         6 |
|  2 | LogisticRegression         | 0.4956  | 0.491122 |   0.507177 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.7909  | 0.783336 |   0.690355 |            4 |            5 |         5 |
|  4 | DecisionTreeClassifier     | 0.54185 | 0.563235 |   0.553398 |            7 |            7 |         7 |
|  5 | SVC                        | 0.9136  | 0.914609 |   0.827586 |            1 |            1 |         1 |
|  6 | RandomForestClassifier     | 0.8447  | 0.865273 |   0.75     |            2 |            2 |         2 |
|  7 | LGBMClassifier             | 0.799   | 0.801964 |   0.748858 |            3 |            3 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.5127685409487388, loss='deviance',
                           max_depth=9, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=18, presort='deprecated',
                           random_state=466, subsample=1.0, tol=1e-07,
                           validation_fraction=0.04, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=6.309215587736547, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.2071864720084633, eval_metric='logloss', gamma=0.5,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.207186475, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=39, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=466,
              reg_alpha=6.30921555, reg_lambda=1.4531785778660498e-05,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00016841124758275325, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=466, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=34, p=3,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=11, min_samples_split=16,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=466, splitter='best')
SVC(C=5.432907893494134, break_ties=False, cache_size=200, class_weight=None,
    coef0=9.8, decision_function_shape='ovr', degree=2, gamma='auto',
    kernel='poly', max_iter=-1, probability=True, random_state=466,
    shrinking=True, tol=7.885674696912461e-05, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=9,
                       min_weight_fraction_leaf=0.0, n_estimators=84,
                       n_jobs=None, oob_score=False, random_state=466,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=7,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=80,
               n_jobs=-1, num_leaves=249, objective='binary', random_state=466,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='SFLKXGDR_0.136_0.741_466-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='SFLKXGDR_0.136_0.741_466-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='SFLKXGDR_0.136_0.741_466-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='SFLKXGDR_0.136_0.741_466-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='SFLKXGDR_0.136_0.741_466-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='SFLKXGDR_0.136_0.741_466-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/SFLKXGDR_0.136_0.741_466.html)