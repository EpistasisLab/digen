# Dataset SXLGKFDR_0.165_0.8_5390

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.8864  | 0.875314 |   0.79803  |            4 |            4 |         4 |
|  1 | XGBClassifier              | 0.9144  | 0.905764 |   0.84058  |            2 |            2 |         3 |
|  2 | LogisticRegression         | 0.5239  | 0.568406 |   0.507463 |            8 |            7 |         8 |
|  3 | KNeighborsClassifier       | 0.8529  | 0.863168 |   0.746411 |            5 |            5 |         5 |
|  4 | DecisionTreeClassifier     | 0.52575 | 0.548022 |   0.590717 |            7 |            8 |         7 |
|  5 | SVC                        | 0.98175 | 0.969314 |   0.950495 |            1 |            1 |         1 |
|  6 | RandomForestClassifier     | 0.8171  | 0.799915 |   0.729064 |            6 |            6 |         6 |
|  7 | LGBMClassifier             | 0.9005  | 0.894575 |   0.841584 |            3 |            3 |         2 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.09780882398086349, loss='deviance',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=33, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=16, presort='deprecated',
                           random_state=5390, subsample=1.0, tol=1e-07,
                           validation_fraction=0.03, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.005077267511381433, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.5089011501157032, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.508901179, max_delta_step=0, max_depth=7,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=98, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=5390,
              reg_alpha=0.00507726753, reg_lambda=97.44416625514664,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00026438952153928617, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5390, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=40, p=3,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=9, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=16, min_samples_split=9,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=5390, splitter='best')
SVC(C=69781.4650452434, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=1.5, decision_function_shape='ovr', degree=2,
    gamma='scale', kernel='poly', max_iter=-1, probability=True,
    random_state=5390, shrinking=True, tol=1.9341262972028122e-05,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=19,
                       min_weight_fraction_leaf=0.0, n_estimators=96,
                       n_jobs=None, oob_score=False, random_state=5390,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=8,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=96,
               n_jobs=-1, num_leaves=72, objective='binary', random_state=5390,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='SXLGKFDR_0.165_0.8_5390-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='SXLGKFDR_0.165_0.8_5390-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='SXLGKFDR_0.165_0.8_5390-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='SXLGKFDR_0.165_0.8_5390-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='SXLGKFDR_0.165_0.8_5390-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='SXLGKFDR_0.165_0.8_5390-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/SXLGKFDR_0.165_0.8_5390.html)