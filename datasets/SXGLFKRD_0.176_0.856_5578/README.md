# Dataset SXGLFKRD_0.176_0.856_5578

|    | classifiers                |   auroc |    auprc |   f1_score |   rank_auroc |   rank_auprc |   rank_f1 |
|---:|:---------------------------|--------:|---------:|-----------:|-------------:|-------------:|----------:|
|  0 | GradientBoostingClassifier | 0.9664  | 0.972265 |   0.909091 |            3 |            3 |         3 |
|  1 | XGBClassifier              | 0.9767  | 0.981904 |   0.915423 |            2 |            2 |         2 |
|  2 | LogisticRegression         | 0.5847  | 0.601417 |   0.56872  |            7 |            7 |         7 |
|  3 | KNeighborsClassifier       | 0.9018  | 0.912709 |   0.819512 |            6 |            6 |         6 |
|  4 | DecisionTreeClassifier     | 0.52785 | 0.559217 |   0.526316 |            8 |            8 |         8 |
|  5 | SVC                        | 0.9909  | 0.994212 |   0.975124 |            1 |            1 |         1 |
|  6 | RandomForestClassifier     | 0.9382  | 0.936954 |   0.882353 |            5 |            5 |         5 |
|  7 | LGBMClassifier             | 0.9635  | 0.969662 |   0.906404 |            4 |            4 |         4 |


<details>
<summary>Parameters of tuned ML methods</summary>


```
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.2687140328549146, loss='deviance',
                           max_depth=6, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=108, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=18, presort='deprecated',
                           random_state=5578, subsample=1.0, tol=1e-07,
                           validation_fraction=0.06999999999999999, verbose=0,
                           warm_start=False)
XGBClassifier(alpha=0.02157745121830786, base_score=0.5, booster='gbtree',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              eta=0.3463217729939356, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints=None,
              learning_rate=0.346321762, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=79, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=5578,
              reg_alpha=0.0215774514, reg_lambda=4.659980629003112,
              scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)
LogisticRegression(C=0.5946347613465647, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=5578, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=29, p=1,
                     weights='distance')
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=7, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=19, min_samples_split=18,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=5578, splitter='best')
SVC(C=217.8574261906646, break_ties=False, cache_size=200,
    class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=2,
    gamma='scale', kernel='poly', max_iter=-1, probability=True,
    random_state=5578, shrinking=True, tol=0.003068509452167849, verbose=False)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=None,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, n_estimators=86,
                       n_jobs=None, oob_score=False, random_state=5578,
                       verbose=0, warm_start=False)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=6,
               metric='binary_logloss', min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=98,
               n_jobs=-1, num_leaves=253, objective='binary', random_state=5578,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
```

</details>

<details>
<summary>Expected performance (100 different random seeds)</summary>
<img src='SXGLFKRD_0.176_0.856_5578-box.svg' width=40% />
</details>

<details>
<summary>Receiver Operating Characteristics (ROC) curve</summary>
<img src='SXGLFKRD_0.176_0.856_5578-roc.svg' width=40% />
</details>

<details>
<summary>Precision-Recall Curve</summary>
<img src='SXGLFKRD_0.176_0.856_5578-prc.svg' width=40% />
</details>

<details>
<summary>Model (GP-tree)</summary>
<img src='SXGLFKRD_0.176_0.856_5578-model.svg' height=10% />
</details>

<details>
<summary>Endpoint histogram</summary>
<img src='SXGLFKRD_0.176_0.856_5578-endpoint.svg' width=40% />
</details>

<details>
<summary>Feature correlations</summary>
<img src='SXGLFKRD_0.176_0.856_5578-corr.svg' width=40% />
</details>

[**Pandas Profiling Report**](https://epistasislab.github.io/digen/profile/SXGLFKRD_0.176_0.856_5578.html)