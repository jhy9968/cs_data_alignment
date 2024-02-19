import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


distance_data = pd.read_csv('C:/Users/hjia0058/Downloads/archive/Distance/distances.txt', delimiter=', ', header=None)

print(distance_data.iloc[:, 1])
plt.plot(distance_data.iloc[:, 1])
plt.show()