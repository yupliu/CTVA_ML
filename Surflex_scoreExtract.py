import pandas as pd
# Extract Surflex scores from a docking output file and return them in dataframe
def extract_surflex_scores(surflex_output_file: str) -> pd.DataFrame:
    scores = []
    names = []
    with open(surflex_output_file, 'r') as file:
        for line in file:
            if line.find("_000") != -1:
                parts = line.split()
                pose_id = parts[0][1:-5]
                score = float(parts[1])
                scores.append(score)
                names.append(pose_id)    
    df = pd.DataFrame({'ID': names, 'Score': scores})
    return df


# Retrieve all files with a specific substring in a directory
# For each file, extract Surflex scores and compile them into a single dataframe and then sort by score
import os
def compile_and_sort_surflex_scores(directory: str, substring: str) -> pd.DataFrame:
    all_scores = pd.DataFrame()    
    for filename in os.listdir(directory):
        if substring in filename and not filename.endswith('.mol2'):
            print(f"Processing file: {filename}")
            file_path = os.path.join(directory, filename)
            df = extract_surflex_scores(file_path)
            all_scores = pd.concat([all_scores, df], ignore_index=True)    
    sorted_scores = all_scores.sort_values(by='Score', ascending=False).reset_index(drop=True)
    return sorted_scores

# write a dataframe to a text file without use to_csv
def write_dataframe_to_textfile(df: pd.DataFrame, output_file: str):    
    with open(output_file, 'w') as file:
        file.write('\t'.join(df.columns) + '\n')
        for index, row in df.iterrows():
            file.write('\t'.join(map(str, row.values)) + '\n')


# Test the functions
def test_functions():
    directory = r"/home/am7574/Run_dir/Surflex/Temp"
    substring = "shape"
    sorted_scores_df = compile_and_sort_surflex_scores(directory, substring)
    print(sorted_scores_df.head(4000))
    output_file = r"/home/am7574/Run_dir/Surflex/Temp/compiled_sorted_scores.txt"
    top_df = sorted_scores_df.head(4000)
    write_dataframe_to_textfile(top_df, output_file)
    top_df.to_csv(r"/home/am7574/Run_dir/Surflex/Temp/compiled_sorted_scores.csv", index=False)

# Accept directory and substring as input and run the functions
# -h for help    
if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser(description="Extract and compile Surflex scores from docking/shape output files.")
   parser.add_argument("directory", type=str, help="Directory containing Surflex output files.")
   parser.add_argument("substring", type=str, help="Substring to identify relevant files.")
   args = parser.parse_args()
   sorted_scores_df = compile_and_sort_surflex_scores(args.directory, args.substring)
   print(sorted_scores_df.head(4000))
   output_file = os.path.join(args.directory, "compiled_sorted_scores_" + args.substring + ".txt")
   top_df = sorted_scores_df.head(4000).copy()
   top_df['methods'] = args.substring  # Add methods column with value 'shape'
   write_dataframe_to_textfile(top_df, output_file) 
   top_df.to_csv(os.path.join(args.directory, "compiled_sorted_scores_" + args.substring + ".csv"), index=False)
