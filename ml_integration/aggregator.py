import os
import sys
import csv
from datetime import datetime
import pandas as pd

# Usage: python aggregator.py <parent_directory>

def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregator.py <parent_directory>")
        sys.exit(1)

    parent_dir = sys.argv[1]
    today_str = datetime.now().strftime('%Y-%m-%d')
    folder_path = os.path.join(parent_dir, today_str)
    csv_path = os.path.join(os.path.dirname(__file__), 'masterdata.csv')

    if not os.path.isdir(folder_path):
        print(f"No folder named {today_str} found in {parent_dir}.")
        return

    # Read existing CSV data
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded existing data with {len(df)} rows")
        
        # Find the oldest date in the dataset
        if len(df) > 0:
            # Convert columns to numeric
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
            
            # Create a proper date column for comparison
            df['FullDate'] = pd.to_datetime({
                'year': df['Year'],
                'month': df['Month'], 
                'day': df['Date']
            }, errors='coerce')
            
            # Find the oldest date
            oldest_date = df['FullDate'].min()
            if pd.notna(oldest_date):
                oldest_date_str = oldest_date.strftime('%Y-%m-%d')
                print(f"Oldest date in dataset: {oldest_date_str}")
                
                # Remove all rows with the oldest date
                rows_to_remove = df[df['FullDate'] == oldest_date]
                df_cleaned = df[df['FullDate'] != oldest_date]
                
                print(f"Removed {len(rows_to_remove)} rows with date {oldest_date_str}")
                
                # Drop the temporary FullDate column
                df_cleaned = df_cleaned.drop(columns=['FullDate'])
                
                # Save the cleaned data
                df_cleaned.to_csv(csv_path, index=False)
                print(f"Saved cleaned data with {len(df_cleaned)} rows")
                
                # Now append all CSVs from today's folder
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                if not csv_files:
                    print(f"No CSV files found in {folder_path}")
                else:
                    for csv_file in csv_files:
                        file_path = os.path.join(folder_path, csv_file)
                        temp_df = pd.read_csv(file_path)
                        # Append to masterdata.csv, skip header
                        temp_df.to_csv(csv_path, mode='a', header=False, index=False)
                    print(f"Appended all CSV data from 2025-07-23 to masterdata.csv")
            else:
                print("Could not determine oldest date, keeping existing data")
    else:
        print("No existing masterdata.csv found")

if __name__ == "__main__":
    main()
