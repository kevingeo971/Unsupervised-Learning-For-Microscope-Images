import numpy as np 
import matplotlib.pyplot as plt

X_train_og = np.load("X_train_og.npy")

X_train_lowF = np.load("X_train_lowF.npy")

X_train_highF = np.load("X_train_highF.npy")

X_val_og = np.load("X_val_og.npy")

X_val_lowF = np.load("X_val_lowF.npy")

X_val_highF = np.load("X_val_highF.npy")

n = 8
plt.imsave("X_train_og.png", X_val_og[n])
plt.imsave("X_train_low.png", X_val_lowF[n])
plt.imsave("X_train_high.png", X_val_highF[n])

print("Done")