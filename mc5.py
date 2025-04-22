import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

x = np.random.rand(100).reshape(-1, 1)

y_train = np.array([1 if xi <= 0.5 else 2 for xi in x[:50]])

X_train, X_test = x[:50], x[50:]
y_test = np.array([1 if xi <= 0.5 else 2 for xi in x[50:]])

k_values = [1, 2, 3, 4, 5, 20, 30]

print("\nResults for k-NN classification:")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"k={k}, Accuracy: {acc:.2f}")

plt.figure(figsize=(10, 6))
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    preds = model.predict(x)
    plt.scatter(x[:50], [k] * 50, c=y_train, cmap="viridis", label=f"Training (k={k})" if k == 1 else None)
    plt.scatter(x[50:], [k] * 50, c=preds[50:], cmap="viridis", marker="x")

plt.yticks(k_values, [f"k={k}" for k in k_values])
plt.xlabel("x values")
plt.ylabel("k values")
plt.title("k-NN Classification for Different k Values")
plt.colorbar(label="Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()