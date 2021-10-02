import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

os.chdir('C:/Users/Andrew/Desktop/Datacon/NorthCarolina')

df = pd.read_csv('dataset_covid_final.csv')
df = df.dropna()


### make everything in the same unit (# of people) by multiplying out the percentages with population
df['Percent Uninsured'] = df['Percent Uninsured'].apply(lambda x: int(x.split('%')[0]))
df['Percent Uninsured'] = df['Percent Uninsured']*0.01*df['population']

df['poverty_population'] = df['poverty_population'].apply(lambda x: int(x.replace(",",'')))
df['Unemployment_rate'] = df['Unemployment_rate']*0.01*df['population']
df['Median_Income'] = df['Median_Income'].apply(lambda x: int(x.replace(",",'')))
df['Mask_Never'] = df['Mask_Never']*df['population']
df['Mask_Rarely'] = df['Mask_Rarely']*df['population']
df['Mask_Sometimes'] = df['Mask_Sometimes']*df['population']
df['Mask_Freqeuntly'] = df['Mask_Freqeuntly']*df['population']
df['Mask_Always'] = df['Mask_Always']*df['population']


dataset = df.drop(columns=['fips','State','County','population','Mask_Rarely','Mask_Sometimes','Mask_Freqeuntly'])





### COVID
dataset_covid = df.drop(columns=['fips','State','County','population','Mask_Rarely','Mask_Sometimes','Mask_Freqeuntly','Total_Stroke','Total_Coronary_Heart'])
# sns.pairplot(dataset)

scaler_covid = MinMaxScaler()
scaler_covid.fit(dataset_covid)
dataset_covid = scaler_covid.transform(dataset_covid)


X_covid = dataset_covid[:,:6]
y_covid = dataset_covid[:,-1]


X_train_covid,X_test_covid,y_train_covid,y_test_covid = train_test_split(X_covid,y_covid,test_size=0.2,random_state=0)

xgb_covid = XGBRegressor(verbosity = 0)
xgb_covid.fit(X_train_covid,y_train_covid)
xgb_covid.score(X_train_covid, np.ravel(y_train_covid))
xgb_covid.score(X_test_covid, np.ravel(y_test_covid))
mean_absolute_error(y_train_covid, xgb_covid.predict(X_train_covid))
mean_absolute_error(y_test_covid, xgb_covid.predict(X_test_covid))


# pred = pd.Series(xgb_covid.predict(X))
# y_covid = pd.Series(y_covid)

# plt.figure(figsize=(16, 4))
# plt.plot(y_covid, color='blue', label='COVID Confirmed Case - Actual')
# plt.plot(pred, alpha=0.7, color='red', label='COVID Confirmed Case- Predicted')
# plt.title('COVID - Actual vs. Predicted (MAE = {})'.format(round(mean_absolute_error(y_test, xgb_covid.predict(X_test)),5)))
# plt.xticks([])
# plt.xlabel('County')
# plt.ylabel('Number of COVID Confirmed Case')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

### Stroke
dataset_stroke = df.drop(columns=['fips','State','County','population','Mask_Rarely','Mask_Sometimes','Mask_Freqeuntly','Total_Confirmed','Total_Coronary_Heart'])
#sns.pairplot(dataset)

scaler_stroke = MinMaxScaler()
scaler_stroke.fit(dataset_stroke)
dataset_stroke = scaler_stroke.transform(dataset_stroke)


X = dataset_stroke[:,:6]
y_stroke = dataset_stroke[:,-1]


X_train_stroke,X_test_stroke,y_train_stroke,y_test_stroke = train_test_split(X,y_stroke,test_size=0.2,random_state=0)

xgb_stroke = XGBRegressor(verbosity = 0)
xgb_stroke.fit(X_train_stroke,y_train_stroke)
xgb_stroke.score(X_train_stroke, np.ravel(y_train_stroke))
xgb_stroke.score(X_test_stroke, np.ravel(y_test_stroke))
mean_absolute_error(y_train_stroke, xgb_stroke.predict(X_train_stroke))
mean_absolute_error(y_test_stroke, xgb_stroke.predict(X_test_stroke))


# pred = pd.Series(xgb_stroke.predict(X))
# y_stroke = pd.Series(y_stroke)

# plt.figure(figsize=(16, 4))
# plt.plot(y_stroke, color='blue', label='Total Stroke - Actual')
# plt.plot(pred, alpha=0.7, color='red', label='Total Stroke - Predicted')
# plt.title('Stroke - Actual vs. Predicted (MAE = {})'.format(round(mean_absolute_error(y_test, xgb_stroke.predict(X_test)),4)))
# plt.xticks([])
# plt.xlabel('County')
# plt.ylabel('Number of Stroke Cases')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')


### Coronary Heart
dataset_cor = df.drop(columns=['fips','State','County','population','Mask_Rarely','Mask_Sometimes','Mask_Freqeuntly','Total_Confirmed','Total_Stroke'])
# sns.pairplot(dataset)

scaler_cor = MinMaxScaler()
scaler_cor.fit(dataset_cor)
dataset_cor = scaler_cor.transform(dataset_cor)


X = dataset_cor[:,:6]
y_heart = dataset_cor[:,-1]


X_train_cor,X_test_cor,y_train_cor,y_test_cor = train_test_split(X,y_heart,test_size=0.2,random_state=0)

xgb_cor = XGBRegressor(verbosity = 0)
xgb_cor.fit(X_train_cor,y_train_cor)
xgb_cor.score(X_train_cor, np.ravel(y_train_cor))
xgb_cor.score(X_test_cor, np.ravel(y_test_cor))
mean_absolute_error(y_train_cor, xgb_cor.predict(X_train_cor))
mean_absolute_error(y_test_cor, xgb_cor.predict(X_test_cor))


# pred = pd.Series(xgb_cor.predict(X))
# y_heart = pd.Series(y_heart)

# plt.figure(figsize=(16, 4))
# plt.plot(y_heart, color='blue', label='Total Coronary Heart - Actual')
# plt.plot(pred, alpha=0.7, color='red', label='Total Coronary Heart - Predicted')
# plt.title('Coronary Heart - Actual vs. Predicted (MAE = {})'.format(round(mean_absolute_error(y_test, xgb_cor.predict(X_test)),4)))
# plt.xticks([])
# plt.xlabel('County')
# plt.ylabel('Number of Coronary Heart Cases')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')


import tabpy_client
from tabpy.tabpy_tools.client import Client
client = tabpy_client.Client('http://localhost:9004/')

def predict_covid(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8):
    #_arg1 = state
    #_arg2 = county
    #_arg3 = percent uninsured (in # of people)
    #_arg4 = Mask_Never
    #_arg5 = Mask_Always
    #_arg6 = Poverty_population
    #_arg7 = Unemployment
    #_arg8 = Median_Income
    
    ##test
    #_arg3, _arg4, _arg5, _arg6, _arg7, _arg8 = 50, 30, -10, 40, 40, -10
    
    ## converting percentage input values to decimals
    _arg3 = 1 + (_arg3 * 0.01)
    _arg4 = 1 + (_arg4 * 0.01)
    _arg5 = 1 + (_arg5 * 0.01)
    _arg6 = 1 + (_arg6 * 0.01)
    _arg7 = 1 + (_arg7 * 0.01)
    _arg8 = 1 + (_arg8 * 0.01)
    
    ind = df[df['State']==_arg1][df['County']==_arg2].index
    cur_data = dataset_covid[ind]
    cur_data_inverse = scaler_covid.inverse_transform(cur_data) #converting the scaled values back to actual values
    
    cur_y = cur_data_inverse[0][-1] #separate the y value
    new_X = cur_data_inverse[0][:6].reshape(1,6)
    new_X = new_X * [_arg3, _arg4, _arg5, _arg6, _arg7, _arg8] # applying the input change rates
    new_X = np.append(new_X,cur_y).reshape(1,7)
    
    new_X = scaler_covid.transform(new_X)
    input_X = new_X[0][:6].reshape(1,6)
    
    new_pred = xgb_covid.predict(input_X)
    new_val = np.append(input_X,new_pred).reshape(1,7)
    final_val = int(scaler_covid.inverse_transform(new_val)[0][-1])
    
    return "Before Change: {}, After Change: {}".format(cur_y, final_val)


def predict_stroke(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8):
    #_arg1 = state
    #_arg2 = county
    #_arg3 = percent uninsured (in # of people)
    #_arg4 = Mask_Never
    #_arg5 = Mask_Always
    #_arg6 = Poverty_population
    #_arg7 = Unemployment
    #_arg8 = Median_Income
       
    ## converting percentage input values to decimals
    _arg3 = 1 + (_arg3 * 0.01)
    _arg4 = 1 + (_arg4 * 0.01)
    _arg5 = 1 + (_arg5 * 0.01)
    _arg6 = 1 + (_arg6 * 0.01)
    _arg7 = 1 + (_arg7 * 0.01)
    _arg8 = 1 + (_arg8 * 0.01)
    
    ind = df[df['State']==_arg1][df['County']==_arg2].index
    cur_data = dataset_stroke[ind]
    cur_data_inverse = scaler_stroke.inverse_transform(cur_data) #converting the scaled values back to actual values
    
    cur_y = cur_data_inverse[0][-1] #separate the y value
    new_X = cur_data_inverse[0][:6].reshape(1,6)
    new_X = new_X * [_arg3, _arg4, _arg5, _arg6, _arg7, _arg8] # applying the input change rates
    new_X = np.append(new_X,cur_y).reshape(1,7)
    
    new_X = scaler_stroke.transform(new_X)
    input_X = new_X[0][:6].reshape(1,6)
    
    new_pred = xgb_stroke.predict(input_X)
    new_val = np.append(input_X,new_pred).reshape(1,7)
    final_val = int(scaler_stroke.inverse_transform(new_val)[0][-1])
    
    return "Before Change: {}, After Change: {}".format(cur_y, final_val)


def predict_cor(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8):
    #_arg1 = state
    #_arg2 = county
    #_arg3 = percent uninsured (in # of people)
    #_arg4 = Mask_Never
    #_arg5 = Mask_Always
    #_arg6 = Poverty_population
    #_arg7 = Unemployment
    #_arg8 = Median_Income
    
    ##test
    #_arg3, _arg4, _arg5, _arg6, _arg7, _arg8 = -10, -10, 10, -10, -10, 10
    
    ## converting percentage input values to decimals
    _arg3 = 1 + (_arg3 * 0.01)
    _arg4 = 1 + (_arg4 * 0.01)
    _arg5 = 1 + (_arg5 * 0.01)
    _arg6 = 1 + (_arg6 * 0.01)
    _arg7 = 1 + (_arg7 * 0.01)
    _arg8 = 1 + (_arg8 * 0.01)
    
    ind = df[df['State']==_arg1][df['County']==_arg2].index
    cur_data = dataset_cor[ind]
    cur_data_inverse = scaler_cor.inverse_transform(cur_data) #converting the scaled values back to actual values
    
    cur_y = cur_data_inverse[0][-1] #separate the y value
    new_X = cur_data_inverse[0][:6].reshape(1,6)
    new_X = new_X * [_arg3, _arg4, _arg5, _arg6, _arg7, _arg8] # applying the input change rates
    new_X = np.append(new_X,cur_y).reshape(1,7)
    
    new_X = scaler_cor.transform(new_X)
    input_X = new_X[0][:6].reshape(1,6)
    
    new_pred = xgb_cor.predict(input_X)
    new_val = np.append(input_X,new_pred).reshape(1,7)
    final_val = int(scaler_cor.inverse_transform(new_val)[0][-1])
    
    return "Before Change: {}, After Change: {}".format(cur_y, final_val)


client.deploy('predict_covid','predict_covid','predicted number of COVID cases')
client.deploy('predict_stroke','predict_stroke','predicted number of Stroke cases')
client.deploy('predict_cor','predict_cor','predicted number of Coronary Heart cases')
