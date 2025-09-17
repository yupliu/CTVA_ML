# load a csv file with SMILES strings and convert to sdf file
def convert_smiles_to_sdf(input_csv:str, output_sdf:str):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import PandasTools

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Check if the 'SMILES' column exists
    if 'SMILES' not in df.columns and 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain a SMILES column")

    # Use 'SMILES' or 'smiles' column
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
    # Convert SMILES to RDKit Mol objects
    df['ROMol'] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)

    # Remove rows where conversion failed (i.e., ROMol is None)
    df = df[df['ROMol'].notnull()]

    # Save the DataFrame to an SDF file
    PandasTools.WriteSDF(df, output_sdf, molColName='ROMol', properties=list(df.columns), idName='ID')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert SMILES in a CSV file to an SDF file.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file containing SMILES strings')
    parser.add_argument('output_sdf', type=str, help='Path to the output SDF file')
    args = parser.parse_args()
    convert_smiles_to_sdf(args.input_csv, args.output_sdf)
# Example usage:
# convert_smiles_to_sdf('input.csv', 'output.sdf')
