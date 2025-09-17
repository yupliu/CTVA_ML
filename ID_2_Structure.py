#input a file with a list of molecule IDs with scores and a file with a list of ID and molecule structures
# output a file with a list of molecule IDs with scores and structures

def merge_score_structure(score_file, structure_file, output_file):
    import pandas as pd

    # Read the score file
    score_df = pd.read_csv(score_file)
    
    # Read the structure file
    structure_df = pd.read_csv(structure_file)
    
    # Merge the two dataframes on the 'ID' column
    merged_df = pd.merge(score_df, structure_df, on='ID', how='inner')
    
    # Write the merged dataframe to the output file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")

# Example usage
# merge_score_structure('scores.csv', 'structures.csv', 'merged_output.csv')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Merge score and structure files based on molecule IDs.')
    parser.add_argument('score_file', type=str, help='Path to the score file (CSV format)')
    parser.add_argument('structure_file', type=str, help='Path to the structure file (CSV format)')
    parser.add_argument('output_file', type=str, help='Path to the output file (CSV format)')
    args = parser.parse_args()
    merge_score_structure(args.score_file, args.structure_file, args.output_file)
# Example usage
# merge_score_structure('scores.csv', 'structures.csv', 'merged_output.csv')    