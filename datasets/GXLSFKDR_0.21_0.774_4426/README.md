# Dataset GXLSFKDR_0.21_0.774_4426

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier |  0.9955 | 0.996481 |   0.974874 |            1 |            1 |         1 |
|  1 | XGBClassifier              |  0.9911 | 0.992472 |   0.954774 |            2 |            2 |         2 |
|  2 | LogisticRegression         |  0.4819 | 0.538644 |   0.504854 |            8 |            8 |         7 |
|  3 | KNeighborsClassifier       |  0.5539 | 0.589132 |   0.546341 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     |  0.5045 | 0.572807 |   0.49     |            7 |            7 |         8 |
|  5 | SVC                        |  0.866  | 0.8833   |   0.81592  |            4 |            4 |         4 |
|  6 | RandomForestClassifier     |  0.8223 | 0.841885 |   0.776119 |            5 |            5 |         5 |
|  7 | LGBMClassifier             |  0.9764 | 0.97937  |   0.934673 |            3 |            3 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.6179770379596049, loss='exponential',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=54, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=12, presort='deprecated',
                           random_state=4426, subsample=1.0, tol=1e-07,
                           validation_fraction=0.01, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.41669907637730147, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.43915740020047306, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.439157397, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=88, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=4426,
              reg_alpha=0.416699082, reg_lambda=21.031174639379813,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.3035361938581824, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=4426, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=5,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=9, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=4426, splitter='best')
SVC(C=1002.0135341196037, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3,
    gamma='auto', kernel='poly', max_iter=-1, probability=True,
    random_state=4426, shrinking=True, tol=2.8078677114002388e-05,
    verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=60,
                       n_jobs=None, oob_score=False, random_state=4426,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=9,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
               n_jobs=-1, num_leaves=18, objective='binary', random_state=4426,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='GXLSFKDR_0.21_0.774_4426-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='GXLSFKDR_0.21_0.774_4426-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='GXLSFKDR_0.21_0.774_4426-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='GXLSFKDR_0.21_0.774_4426-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='GXLSFKDR_0.21_0.774_4426-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='GXLSFKDR_0.21_0.774_4426-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/GXLSFKDR_0.21_0.774_4426.html)