import os
import pandas as pd

os.chdir('C:/Users/Andrew/Desktop/Datacon/NorthCarolina/poverty')
df_poverty = pd.DataFrame(columns=['fips2', 'county2', 'ruc_code2','total_est_pct3'])

files = os.listdir()

for file in files:
    if '.csv' not in file: continue
    df = pd.read_csv(file)
    ind = df[df[df.columns[0]].isnull()].index.tolist()
    df_temp = df.iloc[ind[0]+1:ind[1],2:6]
    df_temp = df_temp[1:]
    df_poverty = pd.concat([df_poverty,df_temp])

os.chdir('C:/Users/Andrew/Desktop/Datacon/NorthCarolina/')

df_poverty = df_poverty.dropna(subset=['ruc_code2'])
df_poverty.to_csv('df_poverty_combined.csv',index=False)
