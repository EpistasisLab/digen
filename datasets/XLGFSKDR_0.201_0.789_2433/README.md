# Dataset XLGFSKDR_0.201_0.789_2433

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9866  | 0.98985  |   0.974619 |            3 |            3 |         2 |
|  1 | XGBClassifier              | 0.9967  | 0.996888 |   0.99     |            1 |            1 |         1 |
|  2 | LogisticRegression         | 0.4874  | 0.481774 |   0.47619  |            8 |            8 |         8 |
|  3 | KNeighborsClassifier       | 0.6442  | 0.638204 |   0.553846 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.51625 | 0.521264 |   0.507772 |            7 |            7 |         7 |
|  5 | SVC                        | 0.7812  | 0.766768 |   0.695187 |            5 |            5 |         5 |
|  6 | RandomForestClassifier     | 0.9058  | 0.904805 |   0.806122 |            4 |            4 |         4 |
|  7 | LGBMClassifier             | 0.9907  | 0.99101  |   0.94     |            2 |            2 |         3 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.17875008188577252, loss='deviance',
                           max_depth=6, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=19, presort='deprecated',
                           random_state=2433, subsample=1.0, tol=1e-07,
                           validation_fraction=0.01, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.10214320388518014, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.6860855200494502, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.686085522, max_delta_step=0, max_depth=7,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=53, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=2433,
              reg_alpha=0.102143206, reg_lambda=96.79166393539603,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.00034349714031263093, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=2433, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=8, p=3,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=8, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=17, min_samples_split=5,
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
                       min_samples_leaf=4, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=59,
                       n_jobs=None, oob_score=False, random_state=2433,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=9,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=95,
               n_jobs=-1, num_leaves=127, objective='binary', random_state=2433,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='XLGFSKDR_0.201_0.789_2433-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='XLGFSKDR_0.201_0.789_2433-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='XLGFSKDR_0.201_0.789_2433-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='XLGFSKDR_0.201_0.789_2433-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='XLGFSKDR_0.201_0.789_2433-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='XLGFSKDR_0.201_0.789_2433-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/docs/profile/XLGFSKDR_0.201_0.789_2433.html)