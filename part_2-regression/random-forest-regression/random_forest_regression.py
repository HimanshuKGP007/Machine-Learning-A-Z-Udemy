# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
model = regressor.fit(X, y)

# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(np.array([[6.5]]))

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Feature Importance
importances = model.feature_importances_

#Look for Importances



