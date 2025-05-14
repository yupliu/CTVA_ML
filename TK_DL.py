import rdkit
from rdkit.Chem import PandasTools

tkSDF_File = r"C:\Users\am7574\OneDrive - Corteva\Documents\Projects\AI_chemistry\TK_AI\TK_ALL.sdf"
tkData = PandasTools.LoadSDF(tkSDF_File)
tkDataMol = tkData[["ROMol","IC50 uM"]]
tkDataMol = tkDataMol.dropna()
tkDataMol['activity'] = tkDataMol['IC50 uM'].apply(lambda x: 0 if x == 'NI' or float(x) >= 20 else 1)

def filter_molecules_by_smarts(mols,smarts_pattern):
    from rdkit import Chem
    filtered_mols = []
    for mol in mols:
        if mol is not None and mol.HasSubstructMatch(Chem.MolFromSmarts(smarts_pattern)):
            filtered_mols.append(mol)
    return filtered_mols


import pandas as pd
def calcFingerprint(data: pd.DataFrame) -> pd.DataFrame:
    from rdkit.Chem import rdFingerprintGenerator
    fpGen = rdFingerprintGenerator.GetMorganGenerator()
    data["MolFP"] = data["ROMol"].apply(lambda x:fpGen.GetCountFingerprintAsNumPy(x))
    return data
dataFP = calcFingerprint(tkDataMol)

y = tkDataMol['activity'].to_list()
x = tkDataMol['MolFP'].to_list()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class TKModel(nn.Module):
    def __init__(self, input_dim):
        super(TKModel, self).__init__()
        self.input_dim = input_dim
        self.create_model(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.epochs = 20
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    def create_model(self,input_dim):
        self. model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

model = TKModel(input_dim=len(x_train[0]))
model.to(model.device)

for epoch in range(model.epochs):
    model.train()
    model.optimizer.zero_grad()
    y_pred = model(torch.tensor(x_train).float())
    loss = model.criterion(y_pred, torch.tensor(y_train).float().view(-1,1))
    loss.backward()
    model.optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

from rdkit.Chem import Draw

#draw molecules
def DrawMol(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol)



