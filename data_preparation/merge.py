import os
import pandas as pd

folder_path = r"C:\Users\pc\Desktop\student dataset\subset_with resolution" 

# List all files in the folder
files = os.listdir(folder_path)

# Display all files
for file in files:
    print(file)



csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Read and combine all CSV files into a single DataFrame
df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(r"C:\Users\pc\Desktop\student dataset\student_resolution.csv", index=False)

print("All CSV files have been merged successfully!")
