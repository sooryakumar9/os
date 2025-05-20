import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")

plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names,
filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

print("\nDecision Tree Rules:\n", export_text(model, feature_names=list(data.feature_names)))

sample = np.array([20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812,
    0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134,
    0.01389, 0.003532, 25.38, 24.99, 166.1, 2019.0, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189]).reshape(1, -1)

print(f"\nNew Sample Classification:\nPredicted Class:{data.target_names[model.predict(sample)[0]]}")