# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
temp = np.array(y).reshape((-1, 1))#This was done for converting the vector Y into a matrix for feature scaling
X = sc_X.fit_transform(X)     
y = sc_y.fit_transform(temp)


# Fitting Regression to the dataset - SVR
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
# create your regressor here
# Predicting a new result with SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
       
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
