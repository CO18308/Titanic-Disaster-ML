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


train["Pclass"].value_counts()

def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix = column_name)
    df = pd.concat([df, dummies], axis = 1)
    return df

for column in ["Pclass", "Sex", "Age_categories"]:
    train = create_dummies(train, column)
    test = create_dummies(test, column)

holdout = test
# Use all dummy columns for training.
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_categories_Missing', 'Age_categories_Infant','Age_categories_Child', 'Age_categories_Teenager','Age_categories_Young Adult', 'Age_categories_Adult','Age_categories_Senior']

all_X = train[columns]
all_y = train['Survived']
# Setting a random seed makes the results reproducible.
train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size = 0.20, random_state = 42)
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

accuracy = 100*accuracy_score(test_y, predictions)
accuracy = "{:.2f}".format(accuracy)
print("***************************Prediction Complete!!***************************")

lr = LogisticRegression()
lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[columns])

holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,"Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("prediction_result.csv", index=False)
