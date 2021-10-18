from enum import auto
import numpy as np


x_train = np.load('../test/139x139/x_train.npy')[:, 0, 0, :]
y_train = np.load('../test/139x139/y_train.npy')[:, 0, 0, :]
x_train = np.concatenate([x_train, y_train], axis=-1)[:-3]
y_train = y_train[3:]

x_test = np.load('../test/139x139/x_test.npy')[:, 0, 0, :]
y_test = np.load('../test/139x139/y_test.npy')[:, 0, 0, :]
x_test = np.concatenate([x_test, y_test], axis=-1)[:-3]
y_test = y_test[3:]


print(x_train.shape)
from autosklearn.regression import AutoSklearnRegressor

automl = AutoSklearnRegressor(
    time_left_for_this_task=120, 
    per_run_time_limit=30,
    tmp_folder='/hard/lilu/HRSEPP/tmp/autosklearn_regression_tmp')

automl.fit(x_train, y_train)

print(automl.leaderboard())

y_pred = automl.predict(x_test)

import sklearn
print("Test R2 score:", sklearn.metrics.r2_score(y_test, y_pred))