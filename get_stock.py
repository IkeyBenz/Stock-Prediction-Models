"""
get_stock(symbol: str, update=False) -> pandas.DataFrame

Returns a *pandas DataFrame* object containing the open, close, high, and low of a given stock.
For redundant searches, it stores stock data in the ChartData directory.
For first time searches (ie, the stock's data isn't in ChartData) it uses the worldtradingdata.com API to get it.
To update a stored stock csv, pass update=True.
"""


import pandas as pd
import requests
from os.path import isfile, dirname

DATA_PATH = dirname(".") + 'ChartData'


def get_stock(symbol: str, update=False) -> pd.DataFrame:
    # Try using stored csv file
    if not update and isfile(f'{DATA_PATH}/{symbol}.csv'):
        stock = pd.read_csv(f'{DATA_PATH}/{symbol}.csv', index_col="Date")
        stock.index = pd.to_datetime(stock.index)

    # If no csv file is stored, pull data from worldtradingdata.com
    else:
        API_KEY = open('wtd-api-key.txt').read()

        r = requests.get(
            f'https://www.worldtradingdata.com/api/v1/history?symbol={symbol}&sort=newest&api_token={API_KEY}')

        data = eval(r.text)["history"]

        stock = pd.DataFrame(data)
        stock = stock.T

        stock.index = pd.to_datetime(stock.index)
        stock.index.name = 'Date'

        stock.to_csv(f'{DATA_PATH}/{symbol}.csv')  # Store as csv for later use

    return stock
