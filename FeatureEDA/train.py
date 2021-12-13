import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")


def xgb_model(X_train, y_train, X_val, y_val, X_test, verbose):
    params = {'objective': 'reg:linear',
              'eta': 0.01,
              'max_depth': 6,
              'subsample': 0.6,
              'colsample_bytree': 0.7,
              'eval_metric': 'rmse',
              'seed': random_seed,
              'silent': True,
              }

    record = dict()
    # print(X_train.describe())
    # print(X_train.isnull().any())
    model = xgb.train(params
                      , xgb.DMatrix(X_train, y_train)
                      , 100000
                      , [(xgb.DMatrix(X_train, y_train), 'train'),
                         (xgb.DMatrix(X_val, y_val), 'valid')]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks=[xgb.callback.record_evaluation(record)])

    best_idx = np.argmin(np.array(record['valid']['rmse']))
    val_pred = model.predict(xgb.DMatrix(X_val), ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

    return {'val': val_pred, 'test': test_pred, 'error': record['valid']['rmse'][best_idx],
            'importance': [i for k, i in model.get_score().items()]}


def lgb_model(X_train, y_train, X_val, y_val, X_test, verbose):
    params = {'objective': 'regression',
              'num_leaves': 30,
              'min_data_in_leaf': 20,
              'max_depth': 9,
              'learning_rate': 0.004,
              # 'min_child_samples':100,
              'feature_fraction': 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              'lambda_l1': 0.2,
              "bagging_seed": random_seed,
              "metric": 'rmse',
              # 'subsample':.8,
              # 'colsample_bytree':.9,
              "random_state": random_seed,
              "verbosity": -1}

    record = dict()
    model = lgb.train(params
                      , lgb.Dataset(X_train, y_train)
                      , num_boost_round=100000
                      , valid_sets=[lgb.Dataset(X_val, y_val)]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks=[lgb.record_evaluation(record)]
                      )
    # print(record)
    best_idx = np.argmin(np.array(record['valid_0']['rmse']))

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return {'val': val_pred, 'test': test_pred, 'error': record['valid_0']['rmse'][best_idx],
            'importance': model.feature_importance('gain')}


def cat_model(X_train, y_train, X_val, y_val, X_test, verbose):
    model = CatBoostRegressor(iterations=100000,
                              learning_rate=0.004,
                              depth=5,
                              eval_metric='RMSE',
                              colsample_bylevel=0.8,
                              random_seed=random_seed,
                              bagging_temperature=0.2,
                              metric_period=None,
                              early_stopping_rounds=200)
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              use_best_model=True,
              verbose=False)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    # print(model.get_best_score())
    # breakpoint()
    return {'val': val_pred, 'test': test_pred,
            'error': model.get_best_score()['validation']['RMSE'],
            'importance': model.get_feature_importance()}


if __name__ == '__main__':
    X_train_p = pd.read_csv('../data/prediction/X_train.csv')
    X_test = pd.read_csv('../data/prediction/X_test.csv')
    y_train_p = pd.read_csv('../data/prediction/y_train.csv')
    random_seed = 2021
    k = 10
    fold = list(KFold(k, shuffle=True, random_state=random_seed).split(X_train_p))
    np.random.seed(random_seed)

    result_dict = dict()
    val_pred = np.zeros(X_train_p.shape[0])
    test_pred = np.zeros(X_test.shape[0])
    final_err = 0
    verbose = False

    for i, (train, val) in enumerate(fold):
        print(i + 1, "fold.    RMSE")

        X_train = X_train_p.loc[train, :]
        y_train = y_train_p.loc[train, :].values.ravel()
        X_val = X_train_p.loc[val, :]
        y_val = y_train_p.loc[val, :].values.ravel()

        fold_val_pred = []
        fold_test_pred = []
        fold_err = []

        # """ xgboost
        start = datetime.now()
        result = xgb_model(X_train, y_train, X_val, y_val, X_test, verbose)
        fold_val_pred.append(result['val'] * 0.2)
        fold_test_pred.append(result['test'] * 0.2)
        fold_err.append(result['error'])
        print("xgb model.", "{0:.5f}".format(result['error']),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # """ lightgbm
        start = datetime.now()
        result = lgb_model(X_train, y_train, X_val, y_val, X_test, verbose)
        fold_val_pred.append(result['val'] * 0.4)
        fold_test_pred.append(result['test'] * 0.4)
        fold_err.append(result['error'])
        print("lgb model.", "{0:.5f}".format(result['error']),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # """ catboost model
        start = datetime.now()
        result = cat_model(X_train, y_train, X_val, y_val, X_test, verbose)
        fold_val_pred.append(result['val'] * 0.4)
        fold_test_pred.append(result['test'] * 0.4)
        fold_err.append(result['error'])
        print("cat model.", "{0:.5f}".format(result['error']),
              '(' + str(int((datetime.now() - start).seconds)) + 's)')
        # """

        # mix result of multiple models
        val_pred[val] += np.sum(np.array(fold_val_pred), axis=0)
        print(fold_test_pred)
        test_pred += np.sum(np.array(fold_test_pred), axis=0) / k
        final_err += (sum(fold_err) / len(fold_err)) / k

        print("---------------------------")
        print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))
        print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.sum(np.array(fold_val_pred), axis=0) - y_val) ** 2))))

        print('')

    print("final avg   err.", final_err)
    print("final blend err.", np.sqrt(np.mean((val_pred - y_train_p.values.ravel()) ** 2)))

    sub = pd.read_csv('./sample_submission.csv')
    df_sub = pd.DataFrame()
    df_sub['id'] = sub['id']
    df_sub['revenue'] = np.expm1(test_pred)
    print(df_sub['revenue'])
    df_sub.to_csv('./submission.csv', index=False)
