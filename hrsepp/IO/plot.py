import numpy as np
import matplotlib.pyplot as plt

y_train_pred_1 = np.load('/hard/lilu/y_train_pred_1.npy')
y_test_pred_1 = np.load('/hard/lilu/y_test_pred_1.npy')
y_valid_pred_1 = np.load('/hard/lilu/y_valid_pred_1.npy')

y_train_obs_1 = np.load('/hard/lilu/y_train_obs_1.npy')
y_test_obs_1 = np.load('/hard/lilu/y_test_obs_1.npy')
y_valid_obs_1 = np.load('/hard/lilu/y_valid_obs_1.npy')

r2_train = np.full((7, 224, 224), np.nan)
r2_valid = np.full((7, 224, 224), np.nan)
r2_test = np.full((7, 224, 224), np.nan)

r_train = np.full((7, 224, 224), np.nan)
r_test = np.full((7, 224, 224), np.nan)
r_valid = np.full((7, 224, 224), np.nan)

rmse_train = np.full((7, 224, 224), np.nan)
rmse_test = np.full((7, 224, 224), np.nan)
rmse_valid = np.full((7, 224, 224), np.nan)

from sklearn.metrics import r2_score, mean_squared_error

for i in range(1):
    for j in range(112):
        for k in range(112):

            r2_train[i, j, k] = r2_score(y_train_obs_1[:, i, j, k, 0],
                                         y_train_pred_1[:, i, j, k, 0])
            r2_test[i, j, k] = r2_score(y_test_obs_1[:, i, j, k, 0],
                                        y_test_pred_1[:, i, j, k, 0])
            r2_valid[i, j, k] = r2_score(y_valid_obs_1[:, i, j, k, 0],
                                         y_valid_pred_1[:, i, j, k, 0])

            r_train[i, j, k] = np.corrcoef(y_train_obs_1[:, i, j, k, 0],
                                           y_train_pred_1[:, i, j, k, 0])[0, 1]
            r_test[i, j, k] = np.corrcoef(y_test_obs_1[:, i, j, k, 0],
                                          y_test_pred_1[:, i, j, k, 0])[0, 1]
            r_valid[i, j, k] = np.corrcoef(y_valid_obs_1[:, i, j, k, 0],
                                           y_valid_pred_1[:, i, j, k, 0])[0, 1]

            rmse_train[i, j, k] = np.sqrt(
                mean_squared_error(y_train_obs_1[:, i, j, k, 0],
                                   y_train_pred_1[:, i, j, k, 0]))
            rmse_test[i, j, k] = np.sqrt(
                mean_squared_error(y_test_obs_1[:, i, j, k, 0],
                                   y_test_pred_1[:, i, j, k, 0]))
            rmse_valid[i, j, k] = np.sqrt(
                mean_squared_error(y_valid_obs_1[:, i, j, k, 0],
                                   y_valid_pred_1[:, i, j, k, 0]))

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(r2_test[0, :, :], vmin=0, vmax=1, cmap='jet')
plt.colorbar()

plt.subplot(3, 2, 2)
plt.imshow(r_test[0, :, :], vmin=0, vmax=1, cmap='jet')
plt.colorbar()

plt.subplot(3, 2, 3)
plt.imshow(rmse_test[0, :, :], vmin=0, vmax=0.04, cmap='jet')
plt.colorbar()

plt.subplot(3, 2, 4)
plt.imshow(np.nanmean(y_test_obs_1[:, 0, :, :, 0], axis=0),
           vmin=0,
           vmax=0.3,
           cmap='jet')
plt.colorbar()

plt.subplot(3, 2, 5)
plt.imshow(np.nanmean(y_test_pred_1[:, 0, :, :, 0], axis=0),
           vmin=0,
           vmax=0.3,
           cmap='jet')
plt.colorbar()

plt.savefig('1.pdf')