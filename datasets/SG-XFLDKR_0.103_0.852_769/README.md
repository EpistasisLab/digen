# Dataset SG-XFLDKR_0.103_0.852_769

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier |  0.9208 | 0.914676 |   0.854369 |            2 |            2 |         2 |
|  1 | XGBClassifier              |  0.9206 | 0.910583 |   0.825871 |            2 |            3 |         4 |
|  2 | LogisticRegression         |  0.6043 | 0.541943 |   0.546448 |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       |  0.8001 | 0.768286 |   0.767857 |            7 |            7 |         7 |
|  4 | DecisionTreeClassifier     |  0.8395 | 0.855359 |   0.80203  |            6 |            6 |         6 |
|  5 | SVC                        |  0.9378 | 0.92365  |   0.88     |            1 |            1 |         1 |
|  6 | RandomForestClassifier     |  0.9119 | 0.906807 |   0.830918 |            4 |            4 |         3 |
|  7 | LGBMClassifier             |  0.8783 | 0.877689 |   0.80597  |            5 |            5 |         5 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.24090050510576627, loss='deviance',
                           max_depth=9, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=4, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=16, presort='deprecated',
                           random_state=769, subsample=1.0, tol=1e-07,
                           validation_fraction=0.03, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.08225330455360111, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.2521605158639428, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.252160519, max_delta_step=0, max_depth=9,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=95, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=769,
              reg_alpha=0.0822533071, reg_lambda=2.579239280862067,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00046378320085688825, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=769, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=18, p=2,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=10, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=9, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=769, splitter='best')
SVC(C=33150.112838183886, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=8.8, decision_function_shape='ovr', degree=2,
    gamma='scale', kernel='poly', max_iter=-1, probability=True,
    random_state=769, shrinking=True, tol=2.59104946511308e-05, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=92,
                       n_jobs=None, oob_score=False, random_state=769,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=7,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=89,
               n_jobs=-1, num_leaves=134, objective='binary', random_state=769,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='SG-XFLDKR_0.103_0.852_769-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='SG-XFLDKR_0.103_0.852_769-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='SG-XFLDKR_0.103_0.852_769-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='SG-XFLDKR_0.103_0.852_769-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='SG-XFLDKR_0.103_0.852_769-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='SG-XFLDKR_0.103_0.852_769-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/SG-XFLDKR_0.103_0.852_769.html)