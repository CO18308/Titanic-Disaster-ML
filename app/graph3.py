import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

test = pd.read_csv(r"F:\project\titanicML\app\test.csv")
train = pd.read_csv(r"F:\project\titanicML\app\train.csv")

def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = \
    pd.cut(df["Age"], cut_points, labels = label_names)
    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", "Infant", "Child","Teenager", "Young Adult", "Adult", "Senior"]
train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)
# Create a spreadsheet-style pivot table as a dataframe.
pivot = train.pivot_table(index = "Age_categories", values = 'Survived')
pivot.plot.bar()
plt.show()
