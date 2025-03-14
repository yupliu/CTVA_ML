import sklearn as sl
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

from sklearn import datasets


db = fetch_california_housing(as_frame=True)
x = db.data
y = db.target

data_x, data_y = fetch_california_housing(return_X_y=True)

def calcCorr(col_ftr):
    corr_matrix = col_ftr.corr()
    return corr_matrix['MedHouseVal'].sort_values(ascending=False)

data = db.frame

cor_ref = calcCorr(data)
print(cor_ref)

plt.scatter(data['MedInc'],data['MedHouseVal'],alpha=0.1)

plt.hist(data['MedInc'])

from sklearn.model_selection import StratifiedShuffleSplit
SSSplit = StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=0)
HouseSplit = SSSplit.split(x,y)
#x_train = []
#x_test = []

for x_train_idx, x_test_idx in HouseSplit:
    print(x_train_idx,x_test_idx)

