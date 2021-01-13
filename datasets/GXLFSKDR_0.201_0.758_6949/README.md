# Dataset GXLFSKDR_0.201_0.758_6949

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9941  | 0.994249 |   0.97     |            1 |            1 |         1 |
|  1 | XGBClassifier              | 0.983   | 0.986621 |   0.96     |            2 |            2 |         2 |
|  2 | LogisticRegression         | 0.4478  | 0.46562  |   0.468293 |            8 |            8 |         7 |
|  3 | KNeighborsClassifier       | 0.6352  | 0.669344 |   0.572973 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.47385 | 0.501354 |   0.433862 |            7 |            7 |         8 |
|  5 | SVC                        | 0.82025 | 0.802764 |   0.783019 |            5 |            5 |         4 |
|  6 | RandomForestClassifier     | 0.8264  | 0.819932 |   0.764706 |            4 |            4 |         5 |
|  7 | LGBMClassifier             | 0.8822  | 0.890648 |   0.784689 |            3 |            3 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.6774438930424329, loss='deviance',
                           max_depth=9, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=62, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=19, presort='deprecated',
                           random_state=6949, subsample=1.0, tol=1e-07,
                           validation_fraction=0.11, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.006418900167518711, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.5283046035079524, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.528304577, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=82, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=6949,
              reg_alpha=0.0064189001, reg_lambda=38.861919076814566,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.04935155299989095, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=6949, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=16, p=1,
                     weights='uniform')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=10, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=6949, splitter='best')
SVC(C=22.456907577256228, break_ties=False, cache_size=200, class_weight=None,
    coef0=8.0, decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='poly', max_iter=-1, probability=True, random_state=6949,
    shrinking=True, tol=1.9208180553186774e-05, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=86,
                       n_jobs=None, oob_score=False, random_state=6949,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=8,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
               n_jobs=-1, num_leaves=237, objective='binary', random_state=6949,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='GXLFSKDR_0.201_0.758_6949-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='GXLFSKDR_0.201_0.758_6949-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='GXLFSKDR_0.201_0.758_6949-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='GXLFSKDR_0.201_0.758_6949-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='GXLFSKDR_0.201_0.758_6949-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='GXLFSKDR_0.201_0.758_6949-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/GXLFSKDR_0.201_0.758_6949.html)