import pandas as pd 


df =pd.read_csv(r"C:\Users\pc\Desktop\student dataset\augmented dataset\full_student_complaints (1).csv")


data_augmentation=pd.read_csv(r"C:\Users\pc\Desktop\student dataset\augmented dataset\5000_student_complaints.csv")

enhanced=pd.read_csv(r"C:\Users\pc\Desktop\student dataset\augmented dataset\NEW_enhanced_student_complaints.csv")
print(df.duplicated().sum())

print(df.columns)

columns=['Genre', 'Reports', 'Age', 'GPA', 'Year', 'Count', 'Gender','Nationality', 'Resolution', 'Student Name', 'Ticket ID']


print("the duplicated columns\n"+df[["Reports","Resolution"]].apply(lambda x:x.duplicated().sum()).to_string())

print("for data augmentation")
print("the duplicated columns\n"+data_augmentation[["Reports","Resolution"]].apply(lambda x:x.duplicated().sum()).to_string())

print("synthical data")
print("the duplicated columns\n"+enhanced[["Reports","Resolution"]].apply(lambda x:x.duplicated().sum()).to_string())
print("duplicate row",enhanced.duplicated().sum())

print(enhanced.duplicated().sum())
