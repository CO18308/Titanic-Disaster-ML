import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

test = pd.read_csv(r"F:\project\titanicML\app\test.csv")
train = pd.read_csv(r"F:\project\titanicML\app\train.csv")

survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha = 0.5, color = 'red', bins = 50)
died["Age"].plot.hist(alpha = 0.5, color = 'blue', bins = 50)
plt.legend(['Survived', 'Died'])
plt.show()
