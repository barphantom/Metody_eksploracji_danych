import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Wczytanie danych
dane = pd.read_csv("dane.txt", delimiter="\t")
print(dane)

scatter_matrix(dane, figsize=(20, 20), diagonal='kde')
plt.tight_layout()
plt.savefig("wykres.png", dpi=300, transparent=None, bbox_inches='tight', pad_inches=0.1)
plt.show()