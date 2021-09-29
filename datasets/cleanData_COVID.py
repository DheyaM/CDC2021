import pandas as pd
import os


home = 'C:/Users/Andrew/Desktop/Datacon/NorthCarolina'
os.chdir(home)

df_covid = pd.read_csv('us-counties.csv')
drop_states = ['Northern Mariana Islands','American Samoa','Guam','Puerto Rico','Virgin Islands']
df_covid = df_covid.drop(df_covid[df_covid['state'].isin(drop_states)].index)

df_total = pd.DataFrame()
states = df_covid['state'].unique()


i = 0
for state in states:
    df_state = df_covid[df_covid['state']==state]
    counties = df_state['county'].unique()
    for county in counties:
        if county == 'Unknown': continue
        temp = df_covid[df_covid['county']==county][df_covid['state']==state]
        if len(temp) == 0: continue
        case_num = sum(temp['cases'])
        death_num = sum(temp['deaths'])
        
        df_total.loc[i,'fips'] = temp['fips'].unique()[0]
        df_total.loc[i,'County'] = county
        df_total.loc[i,'State'] = state
        df_total.loc[i,'Total_Confirmed'] = case_num
        df_total.loc[i,'Total_Deaths'] = death_num
        i += 1

df_total.to_csv('Total_COVID_Count_by_county2.csv',index=False)

