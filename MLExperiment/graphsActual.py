import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def generate_heatmaps(csv_file_path):
    """
    Reads parking data from a CSV file and generates a heatmap for each day of the week.

    The heatmap shows parking spot occupancy (red for occupied, green for vacant)
    with spot IDs on the y-axis and the hour of the day on the x-axis.

    Args:
        csv_file_path (str): The path to the input CSV file.
                             The file must contain 'SpotID', 'Status', and columns for
                             date/time components ('Year', 'Month', 'Date', 'Hour', 'Minute', 'Second').
    """
    try:
        df = pd.read_csv(csv_file_path)
        print("CSV file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    
    df.rename(columns={
        'SpotID': 'spot_id',
        'Status': 'status',
        'Year': 'year',
        'Month': 'month',
        'Date': 'day',
        'Hour': 'hour',
        'Minute': 'minute',
        'Second': 'second'
    }, inplace=True)

   
    try:
        time_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']
        df['timestamp'] = pd.to_datetime(df[time_cols])
    except Exception as e:
        print(f"Error creating timestamp. Check if date/time columns exist and are correct: {e}")
        return

    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour_of_day'] = df['timestamp'].dt.hour

    day_order = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday'
    ]

    for day in day_order:
        print(f"Generating heatmap for {day}...")

        day_df = df[df['day_of_week'] == day]

        if day_df.empty:
            print(f"No data available for {day}. Skipping.")
            continue

        heatmap_data = day_df.pivot_table(
            index='spot_id',
            columns='hour_of_day',
            values='status',
            aggfunc=lambda x: x.mode()[0] if not x.mode().empty else 0
        )

        plt.style.use('dark_background')
        plt.figure(figsize=(20, 12))

        cmap = mcolors.ListedColormap(['#2ca02c', '#d62728']) 

        ax = sns.heatmap(
            heatmap_data,
            cmap=cmap,
            linewidths=.5,
            linecolor='black',
            cbar=False 
        )

        plt.title(f'Parking Occupancy for {day}', fontsize=20, pad=20, color='white')
        plt.xlabel('Hour of the Day', fontsize=14, labelpad=15, color='white')
        plt.ylabel('Spot ID', fontsize=14, labelpad=15, color='white')

        plt.xticks(rotation=0, ha='center', color='white')
        plt.yticks(rotation=0, color='white')
        
        legend_patches = [
            plt.Rectangle((0,0),1,1, color='#2ca02c', label='Vacant (0)'),
            plt.Rectangle((0,0),1,1, color='#d62728', label='Occupied (1)')
        ]
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.show()
        print(f"Displayed heatmap for {day}.")


if __name__ == '__main__':
    csv_file = 'master_shuffled_data1.csv'
    generate_heatmaps(csv_file)
