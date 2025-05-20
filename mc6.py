import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, x0, tau):
    return np.exp(-np.square(x - x0) / (2 * tau**2))

def lwr_predict(x_train, y_train, x0, tau):
    W = np.diag(gaussian_kernel(x_train, x0, tau))
    X = np.c_[np.ones(len(x_train)), x_train]
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y_train)
    return np.dot([1, x0], theta)

np.random.seed(42)
x_train = np.linspace(0, 10, 100)
y_train = np.sin(x_train) + np.random.normal(0, 0.2, size=100)
x_test = np.linspace(0, 10, 100)

plt.figure(figsize=(12, 8))
for tau in [0.1, 0.5, 1, 5]:
    y_pred = [lwr_predict(x_train, y_train, x, tau) for x in x_test]
    plt.plot(x_test, y_pred, label=f'tau={tau}')
plt.scatter(x_train, y_train, color='black', alpha=0.5, label='Training Data')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()