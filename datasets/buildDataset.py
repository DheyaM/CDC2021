import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('C:/Users/Andrew/Desktop/Datacon/NorthCarolina')


## read the datasets we have
df= pd.read_csv('uninsured_num.csv')
df_covid = pd.read_csv('Total_COVID_Count.csv')
df_mask = pd.read_csv('mask-use-by-county.csv')
df_heart = pd.read_csv('heart disease.csv')
df_poverty = pd.read_csv('df_poverty_combined.csv')
df_unemp = pd.read_csv('df_unemp_combined.csv')

##rename the columsn as needed
df_mask = df_mask.rename(columns={'NEVER':'Mask_Never','RARELY':'Mask_Rarely','SOMETIMES':'Mask_Sometimes','FREQUENTLY':'Mask_Freqeuntly',
                                  'ALWAYS':'Mask_Always'})


## on the heart disease dataset, only pull out 2018 data

## before we combine datasets, check if fips code matches between the datasets
#check fips between uninsured and covid datasets
test1 = df[['State Name', 'FIPS Code', 'County Name']]
test2 = df_covid[['State','fips','County']]

test1 = test1.sort_values(by=['FIPS Code'])
test1['County Name'] = test1['County Name'].apply(lambda x: x.split(' ')[0])
test2 = test2.sort_values(by=['fips'])

test3 = pd.merge(test1,test2,left_on='FIPS Code',right_on='fips')
print(test3['State Name'].compare(test3['State']))
print(test3['FIPS Code'].compare(test3['FIPS Code']))
print(test3['County Name'].compare(test3['County']))

#check fips between covid and heart dataset
test4 = df_heart[['LocationID','LocationDesc']]
test4 = test4.drop_duplicates()
test4=test4.sort_values(by=['LocationID'])
test5 = df_covid[['fips','County']]
test5 = test5.sort_values(by=['fips'])
test6 = pd.merge(test4,test5,left_on='LocationID',right_on='fips')
print(test6['LocationID'].compare(test6['fips']))
print(test6['LocationDesc'].compare(test6['County'])) ##heart disease dataset good to combine!

#check fips between covid and poverty
test7 = df_poverty[['fips2','county2']]
test7 = test7.sort_values(by=['fips2'])
test8 = pd.merge(test5,test7,left_on='fips',right_on='fips2')
print(test8['fips'].compare(test8['fips2']))
print(test8['County'].compare(test8['county2']))

#check fips between covid and unemployment
test9 = df_unemp[['fips','county']]
test9 = test9.sort_values(by=['fips'])
test10 = pd.merge(test9,test5,on='fips')

## edit heart disease datatset
# extract 2018 data only
df_heart = df_heart[df_heart['Year']=='2018']
df_stroke = df_heart[df_heart['Topic']=='Stroke']
df_cor = df_heart[df_heart['Topic']== 'Coronary Heart Disease']

#add 35 - 64 and 64 and older groups
df_stroke_add = pd.DataFrame()
df_cor_add = pd.DataFrame()
i = 0
for ID in df_stroke['LocationID'].unique():
    temp = df_stroke[df_stroke['LocationID']==ID]
    df_stroke_add.loc[i,'fipID'] = ID
    df_stroke_add.loc[i,'Total_Stroke'] = sum(temp['Data_Value']) * 100000
    i += 1

i = 0
for ID in df_cor['LocationID'].unique():
    temp = df_cor[df_cor['LocationID']==ID]
    df_cor_add.loc[i,'fipID'] = ID
    df_cor_add.loc[i,'Total_Coronary_Heart'] = sum(temp['Data_Value']) * 100000
    i += 1

    

## there are some mismatch on the county names however, it seems like the differences are between shortened name vs. full name
## we will use the full name. Now, combine the datasets
df_merge_ins = df[['State Name', 'FIPS Code', 'County Name','Percent Uninsured']]
df_merge_covid = df_covid[['fips', 'County', 'State', 'Total_Confirmed']]
df_new = pd.merge(df_merge_ins,df_merge_covid,left_on='FIPS Code',right_on='fips')
df_new = pd.merge(df_new,df_mask,left_on='fips',right_on='COUNTYFP')
df_new = pd.merge(df_new,df_stroke_add,left_on='fips',right_on='fipID')
df_new = pd.merge(df_new,df_cor_add,left_on='fips',right_on='fipID')
df_new = pd.merge(df_new,df_poverty,left_on='fips',right_on='fips2')
df_new = pd.merge(df_new,df_unemp,on='fips')
df_new = df_new.drop(columns=['State Name', 'FIPS Code', 'County Name','COUNTYFP','fipID_x','fipID_y','county','county2','ruc_code2','fips2'])
colnames = ['fips','State','County','Percent Uninsured','Mask_Never', 'Mask_Rarely', 'Mask_Sometimes','Mask_Freqeuntly', 'Mask_Always','total_est_pct3','Unemployment_rate','Median_Income','Total_Confirmed','Total_Stroke','Total_Coronary_Heart']
df_new = df_new.reindex(columns=colnames)
df_new = df_new.rename(columns={'total_est_pct3':'poverty_population'})
df_new.to_csv('dataset_covid_4th.csv',index=False)

## dataset update log
#1st = fips, state, county, percent uninsured, total confirmed
#2nd = added mask data
#fips','State','County','Percent Uninsured','Mask_Never', 'Mask_Rarely', 'Mask_Sometimes','Mask_Freqeuntly', 'Mask_Always','Total_Confirmed'
#3rd = added heart disease (stroke and Coronary Heart Disease) data 
#3rd dataset source: https://chronicdata.cdc.gov/browse?category=Heart+Disease+%26+Stroke+Prevention
#4th = added poverty, unemployment rate, income, total stroke cases, total coronary heart disease cases