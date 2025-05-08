import pandas as pd
import os

def merge_xlsx_files(folder_path, output_file=r'C:\Users\pc\Desktop\student dataset\augmented dataset\merged_feedback.xlsx'):
    
    # Get list of all XLSX files in the folder
    xlsx_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    
    if not xlsx_files:
        print("No XLSX files found in the specified folder.")
        return
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    
    for file in xlsx_files:
        file_path = os.path.join(folder_path, file)
        try:
            # Corrected: pd.read_excel() instead of pd.read_xlsx()
            df = pd.read_excel(file_path)
            dfs.append(df)
            print(f"Successfully read {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid XLSX files could be read.")
        return
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True, axis=0)
    
    # Write the merged DataFrame to a new XLSX file
    # Removed os.path.join() since output_file is already a full path
    try:
        # Corrected: pd.DataFrame.to_excel() instead of to_xlsx()
        merged_df.to_excel(output_file, index=False)
        print(f"Merged file saved as: {output_file}")
        print(merged_df.shape)
        print(merged_df.duplicated().sum())
        print(merged_df["Feedback"].isna().sum())
    except Exception as e:
        print(f"Error saving merged file: {e}")


folder_path = r'C:\Users\pc\Desktop\student dataset\feedback_dataset'
merge_xlsx_files(folder_path)