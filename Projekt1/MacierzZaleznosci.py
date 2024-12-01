import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

all_data_csv = pd.read_csv("dane_med_lab1.csv")
scatter_matrix(all_data_csv, figsize=(12, 12), diagonal='kde')
plt.suptitle("Macierz rozrzutu danych")
plt.tight_layout()
plt.show()







