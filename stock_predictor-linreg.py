import pandas as pd
import numpy as np
from get_stock import get_stock
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Instantiate Data Frame:
df = get_stock("MSFT")[::-1]
# Declare X and y values for LinReg:
X = np.array(range(len(df)))
y = df[["close"]]

# Split data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

X_train = np.array(X_train).reshape(-1, 1)  # These three 1d arrays need
X_test = np.array(X_test).reshape(-1, 1)    # to be 2d arrays. So we use
#                                             reshape() to convert them.

# Instantiate the LinReg model:
model = LinearRegression(     # Don't calc y-intercept because we
    fit_intercept=False,      # know it is the first item in X.
)

# Fit model using training data:
model.fit(X_train, y_train)

# Test model for accuracy
y_pred = model.predict(X_test)

model.score(y_pred, y_test)
model.coef_

# Visualize prediction
plt.plot(df.index[:len(X_train)], y_train)
plt.plot(df.index[len(X_train):], y_test)
plt.plot(df.index[len(X_train):], y_pred)
plt.show()  # Pretty aweful
