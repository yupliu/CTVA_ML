import rdkit
from rdkit.Chem import PandasTools
import pandas as pd
import sklearn as sl

BGFile_high = r"C:\Users\am7574\OneDrive - Corteva\Documents\Projects\CDMLG\BG\2024_10_High_Septtr.sdf"
BGMol_high = PandasTools.LoadSDF(BGFile_high)
BGMol_high["activity"] = 1
BGFile_low = r"C:\Users\am7574\OneDrive - Corteva\Documents\Projects\CDMLG\BG\2024_10_Low_Septtr.sdf"
BGMol_low = PandasTools.LoadSDF(BGFile_low)
BGMol_low["activity"] = 0
BGMol_all =  pd.concat([BGMol_high, BGMol_low])
BGMol_all = BGMol_all.sample(frac=1).reset_index(drop=True)
BGMol_all = BGMol_all[["ROMol","activity"]]
BGMol_all = BGMol_all.dropna()

def calcFingerprint(stFile: str) -> pd.DataFrame:
    from rdkit.Chem import PandasTools
    sdData = PandasTools.LoadSDF(stFile)
    from rdkit.Chem import rdFingerprintGenerator
    mfpGen = rdFingerprintGenerator.GetMorganGenerator()
    dataMol = sdData[["ROMol"]]
    dataMol = sdData.dropna()
    dataMol["MolFP"] = sdData["ROMol"].apply(lambda x:mfpGen.GetCountFingerprintAsNumPy(x))
    return dataMol

def calcProp(stFile: str) -> pd.DataFrame:
    from rdkit.Chem import PandasTools
    sdData = PandasTools.LoadSDF(stFile)
    dataMol = sdData[["ROMol"]]
    dataMol = sdData.dropna()
    from rdkit.ML.Descriptors import MoleculeDescriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in rdkit.Chem.Descriptors._descList])
    dataMol["prop"] = sdData["ROMol"].apply(lambda x:calc.CalcDescriptors(x))
    return dataMol

def calcFingerprint(mols: pd.DataFrame) -> pd.DataFrame:
    from rdkit.Chem import rdFingerprintGenerator
    mfpGen = rdFingerprintGenerator.GetMorganGenerator()
    mols["MolFP"] = mols["ROMol"].apply(lambda x:mfpGen.GetCountFingerprintAsNumPy(x))
    return mols

def calcProp(mols: pd.DataFrame) -> pd.DataFrame:
    from rdkit.ML.Descriptors import MoleculeDescriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in rdkit.Chem.Descriptors._descList])
    mols["prop"] = mols["ROMol"].apply(lambda x:calc.CalcDescriptors(x))
    return mols

def assessModel(y_test, y_pred):
    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_pred))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

def getBestModel(X_train, y_train, X_test, y_test):
    from sklearn import pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.svm import SVC
    import numpy as np    
    pipe = pipeline.Pipeline([('classifier', RandomForestClassifier())])
    search_space = [{'classifier': [RandomForestClassifier()],
                     'classifier__n_estimators': [10, 100, 1000],
                     'classifier__max_features': [1, 2, 3]},
                    {'classifier': [LogisticRegression(solver='liblinear')],
                     'classifier__penalty': ['l1', 'l2'],
                     'classifier__C': np.logspace(0, 4, 10)},
                    {'classifier': [GradientBoostingClassifier()],
                     'classifier__n_estimators': [10, 100, 1000],
                     'classifier__learning_rate': [0.001, 0.01, 0.1],
                     'classifier__max_depth': [1, 2, 3]},
                     {'classifier': [AdaBoostClassifier()]},
                        {'classifier': [SVC()],
                        'classifier__C': [0.1, 1, 10],
                        'classifier__gamma': [1, 0.1, 0.01],
                        'classifier__kernel': ['rbf', 'poly', 'sigmoid']},
                    ]
    
    clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
    best_model = clf.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    assessModel(y_test, y_pred)
    return best_model

def getBestfeatures(X, y):
    from sklearn.feature_selection import SelectKBest, f_classif
    X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
    print(X_new.shape)
    return X_new

from sklearn.model_selection import train_test_split
X_fp = calcFingerprint(BGMol_all)
X = X_fp["MolFP"].to_list()
y = BGMol_all['activity'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_model = getBestModel(X_train, y_train, X_test, y_test)
print(best_model.best_estimator_)
getBestfeatures(X, y)


import matplotlib.pyplot as plt
plt.hist(y_test, bins=2)
plt.hist(y_train, bins=2)

X_prop = calcProp(BGMol_all)
X = X_prop["prop"].to_list()
y = BGMol_all['activity'].to_list()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
getBestModel(X_train, y_train, X_test, y_test)  
getBestfeatures(X, y)








