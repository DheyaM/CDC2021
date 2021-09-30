import os
import pandas as pd

os.chdir('C:/Users/Andrew/Desktop/Datacon/NorthCarolina/unemployment')
df_unemp = pd.DataFrame(columns=['fips','name','Unemployment_rate','Median_Income'])

files = os.listdir()

for file in files:
    if '.csv' not in file: continue
    df = pd.read_csv(file)
    df = df.dropna(subset=[df.columns[-1]])
    #ind = df[df[df.columns[0]].isnull()].index.tolist()
    #df_temp = df.iloc[ind[0]+1:ind[1],2:6]
    df_temp = df.iloc[1:,[0,1,15,df.shape[1]-2]]
    df_temp = df_temp.rename(columns={df_temp.columns[0]:'fips',
                              df_temp.columns[1]:'name',
                              df_temp.columns[2]:'Unemployment_rate',
                              df_temp.columns[3]:'Median_Income'})
    df_unemp = pd.concat([df_unemp,df_temp])

os.chdir('C:/Users/Andrew/Desktop/Datacon/NorthCarolina/')
df_unemp['county'] = df_unemp['name'].apply(lambda x: x.split(',')[0])
df_unemp = df_unemp.drop(columns=['name'])
colnames = ['fips','county','Unemployment_rate','Median_Income']
df_unemp = df_unemp.reindex(columns=colnames)
df_unemp.to_csv('df_unemp_combined.csv',index=False)
