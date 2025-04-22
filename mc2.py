import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
corr = df.corr()

print("Correlation Matrix:\n", corr)
plt.figure(figsize=(12, 8))

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("/Users/sooryakumar/Desktop/correlation_heatmap.png")
plt.show()

sns.pairplot(df.sample(1000), diag_kind="kde", corner=True)
plt.suptitle("Pair Plot of California Housing Features", fontsize=16, y=1.02)
plt.savefig("/Users/sooryakumar/Desktop/pairplot.png")
plt.show()