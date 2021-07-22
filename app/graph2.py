import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

test = pd.read_csv(r"F:\project\titanicML\app\test.csv")
train = pd.read_csv(r"F:\project\titanicML\app\train.csv")
# We inspect the dimensions of these two files.

class_pivot = train.pivot_table(index = "Pclass", values = "Survived")
class_pivot.plot.bar()
plt.show()
