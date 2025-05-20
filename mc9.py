import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = fetch_olivetti_faces(shuffle=True, random_state=42)
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = GaussianNB().fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(64, 64), cmap='gray')
    ax.set_title(f"True: {y_test[i]}, Pred: {y_pred[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()