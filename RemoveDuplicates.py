# merge two SD files and remove duplicates based on a specific property (e.g., "ID")
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

def remove_duplicates_by_property(sdfile1: str, sdfile2: str, property_name: str, output_file: str):
    # Load the two SD files into pandas DataFrames
    df1 = PandasTools.LoadSDF(sdfile1)
    df2 = PandasTools.LoadSDF(sdfile2)

    # Concatenate the two DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Display the number of entries before deduplication
    print(f"Number of entries before deduplication: {combined_df.shape[0]}")

    # Display the duplicated entries based on the specified property
    duplicated_entries = combined_df[combined_df.duplicated(subset=[property_name], keep=False)]
    print(f"Duplicated entries based on '{property_name}':")
    print(duplicated_entries)

    # Remove duplicates based on the specified property
    deduplicated_df = combined_df.drop_duplicates(subset=[property_name], keep='first')
    print(f"Number of entries after deduplication: {deduplicated_df.shape[0]}")

    # Display the number of duplicates removed
    duplicates_removed = combined_df.shape[0] - deduplicated_df.shape[0]
    print(f"Number of duplicates removed: {duplicates_removed}")

    # Display removed duplicates
    print("Removed duplicates:")
    print(duplicated_entries.drop_duplicates(subset=[property_name]))       
 
    # Save the deduplicated DataFrame to a new SD file
    PandasTools.WriteSDF(deduplicated_df, output_file, properties=list(deduplicated_df.columns), idName=property_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove duplicates from two SD files based on a specific property.")
    parser.add_argument("sdfile1", type=str, help="Path to the first SD file.")
    parser.add_argument("sdfile2", type=str, help="Path to the second SD file.")
    parser.add_argument("property_name", type=str, help="Property name to identify duplicates (e.g., 'ID').")
    parser.add_argument("output_file", type=str, help="Path to the output SD file.")
    args = parser.parse_args()

    remove_duplicates_by_property(
        sdfile1=args.sdfile1,
        sdfile2=args.sdfile2,
        property_name=args.property_name,
        output_file=args.output_file
    )
# Example usage:
# remove_duplicates_by_property("file1.sdf", "file2.sdf", "ID", "deduplicated_output.sdf")
