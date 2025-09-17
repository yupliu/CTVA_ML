# Read excel file and convert to SDF based on a column with SMILES strings
def convert_xls_to_sdf(input_xls:str, output_sdf:str, smiles_col:str='SMILES'):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import PandasTools

    # Load the Excel file into a DataFrame
    df = pd.read_excel(input_xls)

    # Check if the specified SMILES column exists
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'

    if smiles_col not in df.columns:
        raise ValueError(f"Input Excel must contain a '{smiles_col}' column")

    # Convert SMILES to RDKit Mol objects
    df['ROMol'] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)

    # Remove rows where conversion failed (i.e., ROMol is None)
    df = df[df['ROMol'].notnull()]

    # Save the DataFrame to an SDF file
    PandasTools.WriteSDF(df, output_sdf, molColName='ROMol', properties=list(df.columns), idName='COMPOUND_ID')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert SMILES in an Excel file to an SDF file.')
    parser.add_argument('input_xls', type=str, help='Path to the input Excel file containing SMILES strings')
    parser.add_argument('output_sdf', type=str, help='Path to the output SDF file')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='Name of the column containing SMILES strings (default: SMILES)')
    args = parser.parse_args()
    convert_xls_to_sdf(args.input_xls, args.output_sdf, args.smiles_col)
# Example usage:
# convert_xls_to_sdf('input.xlsx', 'output.sdf', smiles_col='SMILES')
# Example usage:
# convert_xls_to_sdf('input.xlsx', 'output.sdf', smiles_col='SMILES')