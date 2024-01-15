# %%
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# %%
import pandas as pd

# %%
data = pd.read_csv('../train.csv')

# %%
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
# print(x.head())
# print(y.head())
print(x.shape, y.shape)

# %%
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=51)


# %%
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


model_to_set = OneVsRestClassifier(SVC(kernel="poly"))

parameters = {
    "estimator__C": [1,2,4,8],
    "estimator__kernel": ["poly","rbf"],
    "estimator__degree":[1, 2, 3, 4],
}

model_tunning = GridSearchCV(model_to_set, param_grid=parameters,
                             scoring='f1_weighted')

model_tunning.fit(x, y)

print(model_tunning.best_score_)
print(model_tunning.best_params_)


# %%



