# Dataset FX-GLSKDR_0.149_0.884_2433

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9859  | 0.972809 |   0.990099 |            2 |            3 |         1 |
|  1 | XGBClassifier              | 0.9864  | 0.97378  |   0.990099 |            2 |            2 |         1 |
|  2 | LogisticRegression         | 0.5198  | 0.517299 |   0.607692 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.8681  | 0.86207  |   0.776119 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.83865 | 0.860435 |   0.720812 |            7 |            7 |         7 |
|  5 | SVC                        | 0.9024  | 0.909757 |   0.802083 |            5 |            5 |         5 |
|  6 | RandomForestClassifier     | 0.9889  | 0.983952 |   0.990099 |            1 |            1 |         1 |
|  7 | LGBMClassifier             | 0.983   | 0.964051 |   0.990099 |            4 |            4 |         1 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.06206878462045449, loss='deviance',
                           max_depth=7, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=15, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=19, presort='deprecated',
                           random_state=2433, subsample=1.0, tol=1e-07,
                           validation_fraction=0.01, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=1.5272292612852035e-05, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.20665089843378226, eval_metric='logloss', gamma=0.0,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.206650898, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=71, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=2433,
              reg_alpha=1.52722932e-05, reg_lambda=0.00034874156715133627,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=9661.150628015803, class_weight=None, dual=True,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=2433, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=48, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=11, min_samples_split=17,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=2433, splitter='best')
SVC(C=57.32793214900961, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=2.1, decision_function_shape='ovr', degree=3,
    gamma='scale', kernel='poly', max_iter=-1, probability=True,
    random_state=2433, shrinking=True, tol=4.7603309155310765e-05,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=19,
                       min_weight_fraction_leaf=0.0, n_estimators=81,
                       n_jobs=None, oob_score=False, random_state=2433,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=4,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=78,
               n_jobs=-1, num_leaves=142, objective='binary', random_state=2433,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='FX-GLSKDR_0.149_0.884_2433-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='FX-GLSKDR_0.149_0.884_2433-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='FX-GLSKDR_0.149_0.884_2433-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='FX-GLSKDR_0.149_0.884_2433-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='FX-GLSKDR_0.149_0.884_2433-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='FX-GLSKDR_0.149_0.884_2433-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/FX-GLSKDR_0.149_0.884_2433.html)