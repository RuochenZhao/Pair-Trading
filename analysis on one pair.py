#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import datetime
import statsmodels.tsa.stattools as ts
import time
import random
from statsmodels.tsa.stattools import adfuller as adf
import dateutil
from johansen import Johansen
import matplotlib.pyplot as plt
import matplotlib.dates as dates


# In[9]:


data = pd.read_csv('/Users/zhaoruochen/Desktop/Senior/research/proj_data.csv',delimiter=',')


# In[10]:


data['prc']=abs(data['prc'])


# In[4]:


#data['date_num'] = data['date'].apply(lambda d: matplotlib.dates.date2num(datetime.datetime.strptime(d, "%Y-%m-%d")))
#Graph price
def graph_price(X,Y):
    result = pd.merge(X, Y, on='date')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    new_x = dates.datestr2num(result['date'])
    get_ipython().run_line_magic('matplotlib', 'inline')
    # Control the default size of figures in this Jupyter notebook
    get_ipython().run_line_magic('pylab', 'inline')
    pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
    ax=result["prc_x"].plot(grid = True) 
    ax=result["prc_y"].plot(grid = True) 
    ax.set_xticklabels(result['date'])
    ax.set_title('Price')


# In[5]:


X=data.loc[data['permco']==7953]
Y=data.loc[data['permco']==7954]
graph_price(X,Y)


# In[6]:


def graph_calculated_price(a,b):
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    result = pd.merge(X, Y, on='date')
    x1=result['prc_x'] #closing price
    newprc_x=pd.Series()
    newprc_x.set_value(result['date'][0],100)
    for i in range(1,x1.size):
        newprc_x.set_value(result['date'][i],newprc_x[i-1]*(result['ret_x'][i-1]+1))
    y1=result['prc_y'] #closing price
    newprc_y=pd.Series(index=result['date'])
    newprc_y.set_value(result['date'][0],100)
    for i in range(1,y1.size):
        newprc_y.set_value(result['date'][i],newprc_y[i-1]*(result['ret_y'][i-1]+1))
    #new_x = dates.datestr2num(result['date'])
    get_ipython().run_line_magic('matplotlib', 'inline')
    # Control the default size of figures in this Jupyter notebook
    get_ipython().run_line_magic('pylab', 'inline')
    pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
    ax=newprc_x.plot(grid = True) 
    ax=newprc_y.plot(grid = True) 
    #ax.set_xticklabels(result['date'])
    ax.set_title('Caculated Price')


# In[7]:


graph_calculated_price(7953,7954)


# In[8]:


#Graph ratio
def graph_ratio(a,b):
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    result = pd.merge(X, Y, on='date')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    get_ipython().run_line_magic('matplotlib', 'inline')
    # Control the default size of figures in this Jupyter notebook
    get_ipython().run_line_magic('pylab', 'inline')
    pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
    ratio=result['prc_x']/result['prc_y']
    ax=ratio.plot(grid = True) 
    ax.set_xticklabels(result['date'])
    ax.set_title('Ratio')


# In[9]:


graph_ratio(7953,7954)


# In[10]:


data['date'] = pd.to_datetime(data['date'])
#Engle-Granger Test
def EG_coin_result(a,b):  
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    start = datetime.datetime(2005, 6, 30)
    end = datetime.datetime(2009, 12, 17)
    mask = (data['date'] > start) & (data['date'] <= end)
    X=X.loc[mask]
    Y=Y.loc[mask]
    result = pd.merge(X, Y, on='date')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    coin_result = ts.coint(x1, y1)
    print ("Test 1: Engle Granger")
    try:
        coin_result = ts.coint(x1, y1)
        print("p-value: ",coin_result[1])
        print('pair: ',a,'&',b)
        if (coin_result[1]<0.05):
            print("passed")
        else:
            print("not passed")
    except:
        return 0


# In[11]:


#Test one pair - 7953 & 7954
EG_coin_result(7953,7954)


# In[12]:


from statsmodels.tsa.vector_ar.vecm import coint_johansen
def johansen_coin_result(a,b):
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    result = pd.merge(X, Y, on='date')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    df=pd.DataFrame(x1,y1)
    arr=np.stack((x1.values,y1.values),axis=1)
    coin_result =coint_johansen(arr,0,1)
    print ("Test 2: Johansen")
    print("\ntest statistics: r<=0, Crit-90%, Crit-95%, Crit-99%")
    trstat = coin_result.lr1                       # trace statistic
    tsignf = coin_result.cvt                       # critical values
    print("r<=0", trstat[0], tsignf[0])
    print("r<=1", trstat[1], tsignf[1])
    print('pair: ',a,'&',b)
    if (trstat[0]>tsignf[0][1]):
        print("passed")
    else:
        print("not passed")


# In[13]:


johansen_coin_result(7953,7954)


# In[14]:


#Ad-Fuller-Test
def adf_test(a):
    X=data.loc[data['permco']==a]
    x1 = X['prc']
    # perform Augmented Dickey Fuller test
    print('Results of Augmented Dickey-Fuller test:')
    dftest = ts.adfuller(x1,1)
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)
    print(dftest[4])
    if (dftest[0]<dftest[4]['5%']):
        print(a,'passed')
    else:
        print(a,'Not passed')


# In[15]:


#Ad-fuller test implementation
#ratio
adf_test(7953)
adf_test(7954)


# In[90]:


#Trading Strategy: 
#Open a position when prices diverge by two Standard Deviations
#High price: short; Low price: long
#Close position when prices meet again
def trade_stdev(a,b):
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    result = pd.merge(X, Y, on='date')
    result.set_index('date')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    diff = x1-y1
    stdeviation=np.std(diff)
    print(stdeviation)
    interest=0.02 #how to count the interest for shorting? assumptions at the end, will be easy
    #allocate the capital, evenly, according to the , less important
    #trading 
    profit=0
    invested=1000
    if x1.shape[0]<y1.shape[0]:
        min_shape=x1.shape[0]
    else:
        min_shape=y1.shape[0]
    trade=False
    for i in range(0,min_shape-1):
        if abs(x1[i]-y1[i])>stdeviation*2 and not trade:
            trade=True
            if x1[i]>y1[i]:
                short_share=1000/x1[i] #here how many shares do we short?
                short_prc=x1[i]
                short=x1
                long_share=1000/y1[i]
                long_prc=y1[i]
                short=y1
            else:
                short_share=1000/y1[i]
                short_prc=y1[i]
                short=y1
                long_share=1000/x1[i]
                long_prc=x1[i]
                long=x1
            print('short',short_prc)
            print('long',long_prc)
        if x1[i]==y1[i] and trade:
            trade=False
            close=x1[i]
            print(close)
            invested=short_share*(short_prc-close)(1-interest)+long_share*close
            profit=short_share*(short_prc-close)(1-interest)+long_share(close-long_prc)
            print('invested',invested)
            print('profit',profit)
        if trade and i==min_shape-2:
            shorted_prc_now=short[i]
            longed_prc_now=long[i]
            print('short',shorted_prc_now)
            print('long',longed_prc_now)
            profit=short_share*(short_prc-shorted_prc_now)*(1-interest)+long_share*(longed_prc_now-long_prc)
            print(profit)
    return_stock=profit/1000
    print('return',return_stock)
    return return_stock


# In[91]:


trade_stdev(7953,7954)


# In[89]:


#This is an extreme scenario because the stock price of 7953 decreased too much, so shorting earned a lot of money


# In[ ]:


#interest rate
#trading period, pair formation period - 5yr
#what i am doing now: assume 1000 investment
#how many shares to short

