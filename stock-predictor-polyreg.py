from get_stock import get_stock
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta


def predict_stock_polyreg(symbol: str):

    df = get_stock(symbol)

    X = np.array(range(len(df)))
    y = df.close

    a, b, c, d = np.polyfit(X, y, 3)

    def f(x):
        return a*x**3 + b*x**2 + c*x + d

    plt.plot(df.index, y)  # Actual
    plt.plot(df.index, [f(x) for x in X])  # Regression model's interetation

    def predict_x_days_future(days: int):
        X_future = np.array(range(len(df), len(df) + days))
        y_future = [f(x) for x in X_future]

        X_future_dates = [df.index[-1] +
                          timedelta(days=i) for i in range(days)]

        plt.plot(X_future_dates, y_future)

        # The predicted price for this many days in the future
        return y_future[-1]

    predict_x_days_future(365)
    plt.title(symbol)
    plt.show()


for symbol in ["MSFT", "NFLX", "AAPL", "TSLA"]:
    predict_stock_polyreg(symbol)
