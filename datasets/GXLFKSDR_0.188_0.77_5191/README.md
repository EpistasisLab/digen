# Dataset GXLFKSDR_0.188_0.77_5191

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9866  | 0.98739  |   0.955224 |            1 |            1 |         1 |
|  1 | XGBClassifier              | 0.9732  | 0.972239 |   0.91     |            2 |            2 |         2 |
|  2 | LogisticRegression         | 0.5407  | 0.542111 |   0.598291 |            8 |            8 |         5 |
|  3 | KNeighborsClassifier       | 0.64325 | 0.643929 |   0.585859 |            5 |            5 |         6 |
|  4 | DecisionTreeClassifier     | 0.56815 | 0.621075 |   0.507463 |            7 |            6 |         8 |
|  5 | SVC                        | 0.5918  | 0.60719  |   0.527473 |            6 |            7 |         7 |
|  6 | RandomForestClassifier     | 0.89    | 0.869948 |   0.81592  |            4 |            4 |         4 |
|  7 | LGBMClassifier             | 0.9669  | 0.959387 |   0.91     |            3 |            3 |         2 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.48032974115967586, loss='deviance',
                           max_depth=8, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=10, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=17, presort='deprecated',
                           random_state=5191, subsample=1.0, tol=1e-07,
                           validation_fraction=0.03, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.0007869442389502372, base_score=0.5, booster='dart',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.6850175130011922, eval_metric='logloss', gamma=0.0,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.685017526, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=94, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=5191,
              reg_alpha=0.000786944234, reg_lambda=22.310615753774773,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=44.10012251965829, class_weight=None, dual=True,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=5191, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=8, p=5,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=10, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=7, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=5191, splitter='best')
SVC(C=0.5616232751521862, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=7.1000000000000005,
    decision_function_shape='ovr', degree=4, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=5191, shrinking=True,
    tol=0.00016606239335696557, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=52,
                       n_jobs=None, oob_score=False, random_state=5191,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=10,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=94,
               n_jobs=-1, num_leaves=109, objective='binary', random_state=5191,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='GXLFKSDR_0.188_0.77_5191-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='GXLFKSDR_0.188_0.77_5191-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='GXLFKSDR_0.188_0.77_5191-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='GXLFKSDR_0.188_0.77_5191-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='GXLFKSDR_0.188_0.77_5191-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='GXLFKSDR_0.188_0.77_5191-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/GXLFKSDR_0.188_0.77_5191.html)