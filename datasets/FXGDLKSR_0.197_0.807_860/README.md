# Dataset FXGDLKSR_0.197_0.807_860

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.95965 | 0.954753 |   0.960396 |            3 |            2 |         1 |
|  1 | XGBClassifier              | 0.9682  | 0.932416 |   0.95     |            2 |            4 |         3 |
|  2 | LogisticRegression         | 0.4766  | 0.517154 |   0.463054 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.6351  | 0.632201 |   0.582915 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.95935 | 0.966886 |   0.95     |            4 |            1 |         3 |
|  5 | SVC                        | 0.5633  | 0.534708 |   0.544554 |            7 |            7 |         7 |
|  6 | RandomForestClassifier     | 0.9705  | 0.935598 |   0.95098  |            1 |            3 |         2 |
|  7 | LGBMClassifier             | 0.9227  | 0.893138 |   0.872549 |            5 |            5 |         5 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.4471256723270318, loss='exponential',
                           max_depth=8, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=5, presort='deprecated',
                           random_state=860, subsample=1.0, tol=1e-07,
                           validation_fraction=0.060000000000000005, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.9250533969925395, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.16351408380750268, eval_metric='logloss', gamma=0.2,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.163514078, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=70, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=860,
              reg_alpha=0.925053418, reg_lambda=3.4982764200829695,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.1088196125784687, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=860, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=27, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=9, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=15,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=860, splitter='best')
SVC(C=142.82724891234977, break_ties=False, cache_size=200, class_weight=None,
    coef0=5.2, decision_function_shape='ovr', degree=3, gamma='scale',
    kernel='poly', max_iter=-1, probability=True, random_state=860,
    shrinking=True, tol=0.0003047398611430893, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=9,
                       min_weight_fraction_leaf=0.0, n_estimators=98,
                       n_jobs=None, oob_score=False, random_state=860,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=8,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=84,
               n_jobs=-1, num_leaves=247, objective='binary', random_state=860,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='FXGDLKSR_0.197_0.807_860-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='FXGDLKSR_0.197_0.807_860-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='FXGDLKSR_0.197_0.807_860-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='FXGDLKSR_0.197_0.807_860-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='FXGDLKSR_0.197_0.807_860-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='FXGDLKSR_0.197_0.807_860-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/FXGDLKSR_0.197_0.807_860.html)