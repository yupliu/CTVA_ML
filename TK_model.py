import rdkit
from rdkit.Chem import PandasTools

stFile = r"C:\Users\am7574\OneDrive - Corteva\Documents\Projects\AI_chemistry\TK_AI\TK_ALL.sdf"
data = PandasTools.LoadSDF(stFile)
dataMol = data[["ROMol","IC50 uM"]]
dataMol = dataMol.dropna()
dataMol['activity'] = dataMol['IC50 uM'].apply(lambda x: 0 if x == 'NI' or float(x) >= 20 else 1)

from rdkit.Chem import rdFingerprintGenerator
mfpGen = rdFingerprintGenerator.GetMorganGenerator()
dataMol["MolFP"] = dataMol["ROMol"].apply(lambda x:mfpGen.GetCountFingerprintAsNumPy(x))

import sklearn as sl
from sklearn.model_selection import train_test_split
X = dataMol['MolFP'].to_list()
y = dataMol['activity'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def assessModel(y_test, y_pred):
    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_pred))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0) 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

assessModel(y_test, y_pred)

from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
assessModel(y_test, y_pred)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression() 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
assessModel(y_test, y_pred)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
assessModel(y_test, y_pred)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
assessModel(y_test, y_pred)

from sklearn.model_selection import GridSearchCV
from sklearn import pipeline

import numpy as np
pipe = pipeline.Pipeline([('classifier' , RandomForestClassifier())])
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))},
    {'classifier' : [GradientBoostingClassifier()]},
    {'classifier' : [AdaBoostClassifier()]},
    {'classifier' : [SVC()]}
]

gs = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
best_clf = gs.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
assessModel(y_test, y_pred)

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

desc_names = [x[0] for x in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
dataMol["desc"] = dataMol["ROMol"].apply(lambda x:calculator.CalcDescriptors(x))

X1 = dataMol['desc'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=42)
gs = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
best_clf = gs.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
assessModel(y_test, y_pred)
