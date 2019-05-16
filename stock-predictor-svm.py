import pandas as pd
import numpy as np
from get_stock import get_stock
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Get data
df = get_stock("AAPL")[::-1]

X = np.reshape(list(range(len(df))), (len(df), 1))

y = df["close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

X_train = np.array(X_train).reshape(-1, 1)  # These three 1d arrays need
# to be 2d arrays. So we use   # reshape() to convert them.
X_test = np.array(X_test).reshape(-1, 1)

# Instantiate Models:
svr_lin = SVR(kernel="linear", C=1e3)
svr_poly = SVR(kernel="poly", C=1e3, degree=2)
svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)

# Fit models:

svr_lin.fit(X_train, y_train)
svr_poly.fit(X_train, y_train)
svr_rbf.fit(X_train, y_train)

# plt.scatter(X, y, color="black", label="Data")
plt.plot(X_test, svr_rbf.predict(y_test), color="red", label="RBF model")
plt.plot(X_test, svr_lin.predict(y_test), color="green", label="Linear Model")
plt.plot(X_test, svr_poly.predict(y_test),
         color="blue", label="Polynomial Model")

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Support Vector Regression")
plt.legend()
plt.show()
