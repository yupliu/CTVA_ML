import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

from rdkit.Chem import MaeMolSupplier
def read_mae_file(file_path:str):
    """
    Read a Maestro (.mae) file and return a list of RDKit Mol objects.
    
    Args:
        file_path (str): Path to the .mae file.
        
    Returns:
        List[rdkit.Chem.Mol]: List of RDKit Mol objects.
    """
    supplier = MaeMolSupplier(file_path)
    mols = []
    try:
        for mol in supplier:
            if mol is not None:
                mols.append(mol)
    except Exception as e:
        print(f"Error reading .mae file: {e}")
        return []
    return mols

file_path = '/home/am7574/Project_backup/TK/NMT.mae'
molecules = read_mae_file(file_path)
