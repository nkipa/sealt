#!/usr/bin/env python
# coding: utf-8

# # Program to determine the Season (BTC, ETH, ALT)

# In[79]:

import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import inspect
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
import matplotlib.patches as patches
from binance.client import Client
from binance_keys import api_key, secret_key
from datetime import date
from datetime import timedelta  
from datetime import datetime
from matplotlib.patches import FancyBboxPatch


# In[80]:





# In[81]:
st.title('Crypto Season Dashboard [season.py]')

st.sidebar.markdown('# Adjust Data Section')

st.sidebar.info("** If you get an error file not then found Update Price data (below)")
generate_new_file = st.sidebar.button('Update Price data')
lookback = st.sidebar.number_input('How many days of data you want to load?', min_value=14, value=49)
lookback = int(lookback)
#st.sidebar.markdown('## Update price data once per day')

rolling_window = 14
datafile = 'df'+ f'{lookback}' + '_season'
file_generated = False
save_path = ''

# In[82]:

client = Client(api_key, secret_key)

# In[83]:

def getbtc():
    info = client.get_exchange_info()
    symbols = [x['symbol'] for x in info['symbols']]
    relevant=[symbol for symbol in symbols if symbol.endswith('BTC')]
    relevant.sort()
    relevant.extend(["BTCUSDT"])
    #relevant = relevant(columns=relevant.loc[:, "B":"D"].columns)
    return (relevant)

# In[84]:

# reset df1
def resetdf():
    df1=pd.read_csv(save_path + f'{datafile}.csv')
    df1.Time = pd.to_datetime(df1.Time)
    first_column = df1.pop('BTCUSDT')
    #time_column = df1.Time
    df1.insert(1, 'BTCUSDT', first_column)
    df1.columns=df1.columns.str.replace("BTC","")
    df1.columns=df1.columns.str.replace("USDT","BTCUSDT")
    df1 = df1.set_index('Time')
    return(df1)

# In[85]:

def savedf():
    global generate_new_file
    dfs=makedf()
    mergeddf = pd.concat(dict(zip(relevant,dfs)), axis=1)
    closesdf = mergeddf.loc[:,mergeddf.columns.get_level_values(1).isin(['Close'])]
    closesdf.columns = closesdf.columns.droplevel(1)
    df1=closesdf.reset_index()
    df1.to_csv(save_path + f'{datafile}.csv', index=False)
    generate_new_file= False
    return(df1)


# In[ ]:

def getdailydata(symbol):
    frame = pd.DataFrame(client.get_historical_klines(symbol,'1d',f'{lookback+1} days ago UTC'))
    if len(frame)>0:
        frame = frame.iloc[:,:5]
        frame.columns = ['Time','Open','High','Low','Close']
        frame = frame.set_index('Time')
        frame.index = pd.to_datetime(frame.index, unit='ms')
        frame = frame.astype(float)
        return frame

# In[87]:

def makedf():
    dfs = []
    for coin in relevant:
        dfs.append(getdailydata(coin))
    return(dfs)

# In[88]:

# Get all BTC trading pairs from Binance
relevant=getbtc()

# ### Start Program

# In[89]:

# *************** Use to generate new data (DONT USE ALWAYS) ***************
if (generate_new_file):
    df=savedf()

# In[91]:

df=resetdf()
df.reset_index()
df.fillna(1, inplace=True)

# In[92]:

returns = np.log(df / df.shift(1)).dropna()
log_returns = np.log(returns+1)

# In[95]:

dfbtc=log_returns.pop('BTCUSDT')
dfeth=log_returns.pop('ETH')
dfalt=log_returns.mean(axis=1)
dfn=dfbtc.to_frame().join(dfeth)
dfn['ALT']=dfalt
dfn['Season'] = dfn.apply(lambda x: dfn.columns[x.argmax()], axis = 1)

# In[97]:

dfn=dfn.rolling(window=rolling_window, min_periods=1).sum(inplace=True)
dfn['Season'] = dfn.apply(lambda x: dfn.columns[x.argmax()], axis = 1)

# In[100]:

timeplot=dfn[dfn['Season'] != dfn['Season'].shift(1)].index.tolist()

# In[102]:

plt.figure(figsize = (18, 10))
plt.plot(dfn['BTCUSDT'], 'blue')
plt.plot(dfn['ETH'], 'orange')
plt.plot(dfn['ALT'],'green')
plt.legend(['BTCUSDT','ETH','ALT'],loc ="lower right")
plt.vlines(x = timeplot, ymin = dfn.min()[0:3].min(), ymax = dfn.max()[0:3].max(),colors = 'black')
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(axis ='y')

plt.show()

# In[103]:

dfb=[]
dfb=pd.read_csv(save_path + f'{datafile}.csv')
dfb.Time = pd.to_datetime(dfb.Time)
first_column = dfb.pop('BTCUSDT')
time_column = dfb.Time
dfb.insert(1, 'BTCUSDT', first_column)
dfb.columns=dfb.columns.str.replace("BTC ","")
#dfb.columns=dfb.columns.str.replace("USDT","BTCUSDT")
dfb = dfb.set_index('Time')

# In[109]:

dfn['Max']=dfn.iloc[:,:3].max(axis=1)
dfn = dfn[['BTCUSDT', 'ETH', 'ALT',  'Max', 'Season']]

# In[116]:
    
tickers1 = ['BTCUSDT', 'ETH', 'ALT', 'Max'] 
     
dropdown = st.multiselect('Pick your assets', tickers1, default=['BTCUSDT', 'ETH', 'ALT'])    

st.sidebar.markdown('Time Frame ( Display time in days )')
days_to_subtract=st.sidebar.slider('Recommended: 49 days'
                            ,7,lookback, 49,step=7)
st.sidebar.markdown('Rolling Window ( Moving average of daily log returns )')
rolling_window=st.sidebar.slider('Recommended: 14'
                          ,7,lookback,14, step=7) 

d = datetime.today() - timedelta(days=days_to_subtract-1)
#start = st.date_input('Start', value = pd.to_datetime(df.index[0]))
start = st.date_input('Start', value = d)
end = st.date_input('End', value = pd.to_datetime('today'))
   
returns = np.log(df / df.shift(1)).dropna()
log_returns = np.log(returns+1)
dfbtc=log_returns.pop('BTCUSDT')
dfeth=log_returns.pop('ETH')
dfalt=log_returns.mean(axis=1)
dfn=dfbtc.to_frame().join(dfeth)
dfn['ALT']=dfalt
dfn['Season'] = dfn.apply(lambda x: dfn.columns[x.argmax()], axis = 1)
dfs=dfn.copy()

dfn=dfn.rolling(window=rolling_window, min_periods=1).sum(inplace=True)
dfn['Season'] = dfn.apply(lambda x: dfn.columns[x.argmax()], axis = 1)
dfn['Max']=dfn.iloc[:,:3].max(axis=1)
dfn = dfn[['BTCUSDT', 'ETH', 'ALT',  'Max', 'Season']]
df=dfn.copy()

delta=end-start
st.sidebar.subheader(" Actual days displayed: " + str(delta.days+1)+' days') 


# In[117]:

if len(dropdown)>0:
    df = df.loc[str(start):str(end),dropdown]
    dfn = dfn.loc[str(start):str(end),dropdown]
       
    dfn['Season'] = dfn.apply(lambda x: dfn.columns[x.argmax()], axis = 1)    
    dfn['Season'] = dfn['Season'].str.replace('BTCUSDT','Bitcoin')
    dfn['Season'] = dfn['Season'].str.replace('ALT','Alternative Coins')
    dfn['Season'] = dfn['Season'].str.replace('ETH', 'Etherium')
    st.title( 'Seasons now : '+dfn['Season'][-1] + ' Season' )
  
    st.line_chart(df)
   




