# Dataset XLGFSKDR_0.206_0.716_5390

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier |  0.9579 | 0.945125 |   0.945813 |            3 |            3 |         1 |
|  1 | XGBClassifier              |  0.9766 | 0.978372 |   0.929293 |            1 |            1 |         2 |
|  2 | LogisticRegression         |  0.4762 | 0.47175  |   0.458333 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       |  0.53   | 0.658636 |   0.552381 |            6 |            5 |         6 |
|  4 | DecisionTreeClassifier     |  0.5133 | 0.49537  |   0.467005 |            7 |            7 |         7 |
|  5 | SVC                        |  0.5794 | 0.594382 |   0.558824 |            5 |            6 |         5 |
|  6 | RandomForestClassifier     |  0.7312 | 0.664686 |   0.698113 |            4 |            4 |         4 |
|  7 | LGBMClassifier             |  0.9649 | 0.969534 |   0.91     |            2 |            2 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.5291273593372556, loss='deviance',
                           max_depth=8, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=16, presort='deprecated',
                           random_state=5390, subsample=1.0, tol=1e-07,
                           validation_fraction=0.09, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=2.9158551172476994e-05, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.6091679236982254, eval_metric='logloss', gamma=0.0,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.609167933, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=92, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=5390,
              reg_alpha=2.91585511e-05, reg_lambda=43.034014374877444,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=55.86087410148022, class_weight=None, dual=True,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5390, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=14, min_samples_split=16,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=5390, splitter='best')
SVC(C=4666.158831995879, break_ties=False, cache_size=200, class_weight=None,
    coef0=7.1000000000000005, decision_function_shape='ovr', degree=5,
    gamma='auto', kernel='poly', max_iter=-1, probability=True,
    random_state=5390, shrinking=True, tol=0.0002887462566071054,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=9, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=6, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=54,
                       n_jobs=None, oob_score=False, random_state=5390,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=10,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=94,
               n_jobs=-1, num_leaves=158, objective='binary', random_state=5390,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='XLGFSKDR_0.206_0.716_5390-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='XLGFSKDR_0.206_0.716_5390-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='XLGFSKDR_0.206_0.716_5390-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='XLGFSKDR_0.206_0.716_5390-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='XLGFSKDR_0.206_0.716_5390-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='XLGFSKDR_0.206_0.716_5390-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/XLGFSKDR_0.206_0.716_5390.html)