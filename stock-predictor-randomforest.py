import pandas as pd
from get_stock import get_stock
import matplotlib.pyplot as plt

df = get_stock("BTU")

plt.plot(df.index, df.close)
plt.show()
