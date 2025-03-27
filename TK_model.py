import rdkit
from rdkit.Chem import PandasTools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit

stFile = r"C:\Users\am7574\OneDrive - Corteva\Documents\Projects\AI_chemistry\TK_AI\TK_ALL.sdf"
data = PandasTools.LoadSDF(stFile)
dataMol = data[["ROMol","IC50 uM"]]
dataMol = dataMol.dropna()
dataMol['activity'] = dataMol['IC50 uM'].apply(lambda x: 0 if x == 'NI' or float(x) >= 20 else 1)

def assessModel(y_test, y_pred):
    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_pred))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

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

    des_name = [x[0] for x in rdkit.Chem.Descriptors._descList if (x[0].find("PartialCharge") == -1 and x[0].find("BCUT2D")==-1 and x[0].find("Morgan") == -1)]
    #print(des_name)  
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(des_name)
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

def getBestfeatures(X,y):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    X_new = SelectKBest(mutual_info_classif, k=50).fit_transform(X, y)
    print(X_new.shape)
    return X_new

def getBestFeatureSFS(X,y):
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.ensemble import RandomForestClassifier
    sfs_fwd = SequentialFeatureSelector(RandomForestClassifier(), n_features_to_select=30, direction='backward').fit(X, y)
    X_new = sfs_fwd.transform(X)
    print(X_new.shape)
    return X_new


X_fp = calcFingerprint(dataMol)
X = X_fp["MolFP"].to_list()
y = dataMol['activity'].to_list()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
getBestModel(X_train, y_train, X_test, y_test)
X_new = getBestfeatures(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)   
getBestModel(X_train, y_train, X_test, y_test)
""" X_new = getBestFeatureSFS(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)   
getBestModel(X_train, y_train, X_test, y_test) """

X_prop = calcProp(dataMol)
X = X_prop["prop"].to_list()
y = dataMol['activity'].to_list()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
getBestModel(X_train, y_train, X_test, y_test)
X_new = getBestfeatures(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)
getBestModel(X_train, y_train, X_test, y_test)
""" X_new = getBestFeatureSFS(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)   
getBestModel(X_train, y_train, X_test, y_test) """

