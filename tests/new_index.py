import numpy as np
import pandas as pd

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr[:, 1])


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

print(df.values[:, 1])
