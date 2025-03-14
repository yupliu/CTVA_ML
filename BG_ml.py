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
BGMol_all = BGMol_all.sample(frac=1)









