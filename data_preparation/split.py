import pandas as pd
import os

def split_excel_to_csv(excel_file):
    # Define the output folder
    output_folder = r"C:\Users\pc\Desktop\student dataset\subset"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the file (detecting if it's CSV or Excel)
    if excel_file.endswith('.csv'):
        df = pd.read_csv(excel_file)
    elif excel_file.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(excel_file)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # Get total number of rows
    total_rows = df.shape[0]

    # Define size
    chunk_size = 50

    # Create partitions and save to CSV files
    for i in range(0, total_rows, chunk_size):
        partition = df.iloc[i:i + chunk_size]  # Select 50 rows
        output_file = os.path.join(output_folder, f"partition_{i//chunk_size + 1}.csv")
        partition.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

# Example usage
split_excel_to_csv(r"C:\Users\pc\Desktop\student dataset\Datasetprojpowerbi.csv")
