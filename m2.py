import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

correlation_matrix = df.corr()
print("Correlation Matrix:\n")
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("/mnt/data/correlation_heatmap.png")
plt.show()

sns.pairplot(df.sample(1000), diag_kind="kde", corner=True)
plt.suptitle("Pair Plot of California Housing Features", fontsize=16, y=1.02)
plt.savefig("/mnt/data/pairplot.png")
plt.show()