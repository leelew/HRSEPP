import numpy as np


x_train = np.load('x_train.npy')[:, :112, :112]
y_train = np.load('y_train.npy')[:, :112, :112]
x_train = np.concatenate([x_train, y_train], axis=-1)
x_train = x_train[:-3]
y_train = y_train[3:]
x_train[834] = x_train[833]
y_train[831] = y_train[830]
y_train[np.isnan(y_train)] = 0
x_train[np.isnan(x_train)] = 0

print(np.isnan(x_train).any())
print(np.isnan(y_train).any())
np.save('x_train_sub.npy', x_train)
np.save('y_train_sub.npy', y_train)
