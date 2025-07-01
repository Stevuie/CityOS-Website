import pandas as pd
import os
import random

def combine_and_shuffle_csvs(root_folder, output_filename="combined_and_shuffled.csv"):
    all_dataframes = []
    
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path)
                    all_dataframes.append(df)
                    print(f"Successfully read: {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not all_dataframes:
        print(f"No CSV files found in '{root_folder}' or its subfolders.")
        return

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

    try:
        shuffled_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully combined and shuffled all CSVs into '{output_filename}'")
    except Exception as e:
        print(f"Error saving the combined CSV: {e}")

if __name__ == "__main__":
    root_folder_path = "/Users/aryan/Downloads/2025-06-27" 

    output_csv_name = "master_shuffled_data1.csv"
    
    combine_and_shuffle_csvs(root_folder_path, output_csv_name)
