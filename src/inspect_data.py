import pandas as pd

df = pd.read_excel(r'c:\Users\Admin\OneDrive\Desktop\dataset\Online Retail.xlsx', nrows=5)
print(df.columns.tolist())
print(df.head())
