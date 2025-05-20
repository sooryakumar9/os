import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

df_boston = pd.read_csv('BostonHousingDataset.csv')
X, y = df_boston[['RM']], df_boston['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
'''
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Housing Price')
plt.title('Linear Regression on Boston Housing Dataset')
plt.legend()
plt.show()

print(f"Mean Squared Error (Linear Regression): {mean_squared_error(y_test, y_pred):.2f}")
'''
df_auto = pd.read_csv('auto_mpg.csv', na_values='?').dropna()
X, y = df_auto[['horsepower']].astype(float), df_auto['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly_model = make_pipeline(PolynomialFeatures(3), StandardScaler(), LinearRegression())
y_poly_pred = poly_model.fit(X_train, y_train).predict(X_test)

X_sorted, y_sorted = zip(*sorted(zip(X_test.values.ravel(), y_poly_pred)))

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_sorted, y_sorted, color='red', linewidth=2, label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Polynomial Regression on Auto MPG Dataset')
plt.legend()
plt.show()

print(f"Mean Squared (Polynomial Regression): {mean_squared_error(y_test, y_poly_pred):.2f}")