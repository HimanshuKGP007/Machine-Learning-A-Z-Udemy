
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :3].values # .value converts pd series/dataframe to ndarray
Y = dataset.iloc[:, 3].values

# Taking care of missing data
# ---------------------------
from sklearn.impute import SimpleImputer # same logic that of LabelEncoder 
imputer = SimpleImputer(missing_values = np.NaN, strategy='mean')
b = imputer.fit_transform(X[:, 1:3]) 
X[:, 1:3] = b
X

# Encoding categorical data
# -------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # Create Object by calling class -> obj = class()
a = labelencoder_X.fit_transform(X[:,0]) # transform the column 0 and store it in a variable
X[:,0] = a #write on df
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()#Here we need note mention the column because the column is specified in categorical_features
labelencoder_Y = LabelEncoder() # Create Object by calling class -> obj = class()
Y = labelencoder_Y.fit_transform(Y) # transform the column 0 and store it in a variable
Y    #write on df

# Splitting the dataset into the Training set and Test set
# --------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Feature scaling -- Standardisation Method
# ---------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


 
