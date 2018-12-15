#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
#suppress all warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('/Users/zhaoruochen/Desktop/Senior/research/proj_data.csv',delimiter=',')
#set price to be the absolute value
data['prc'] = abs(data.prc)


# In[3]:


#siccd to industry name function
def ffi12(row):
    if (100<=row <=999) or (2000<=row<=2399) or (2700<=row<=2749) or (2770<=row<=2799) or (3100<=row<=3199) or (3940<=row<=3989):
        ffi12='NoDur'
    elif (2500<=row<=2519) or (2590<=row<=2599) or (3630<=row<=3659) or (3710<=row<=3711) or (row==3714) or (row==3716) or (3750<=row<=3751) or (row==3792) or (3900<=row<=3939) or (3990<=row<=3999):
        ffi12='Durbl'
    elif (2520<=row<=2589) or (2600<=row<=2699) or (2750<=row<=2796) or (3000<=row<=3099) or (3200<=row<=3569) or (3580<=row<=3629) or (3700<=row<=3709) or (3712<=row<=3713) or (row==3715) or (3717<=row<=3749) or (3752<=row<=3791) or (3793<=row<=3799) or (3830<=row<=3839) or (3860<=row<=3899):
        ffi12='Manfu'
    elif (1200<=row<=1399) or (2900<=row<=2999):
        ffi12='Enrgy'
    elif (2800<=row<=2829)or (2840<=row<=2899):
        ffi12='Chems'
    elif (3570<=row<=3579) or (3660<=row<=3692) or (3694<=row<=3699) or (3810<=row<=3829) or (7370<=row<=7379):
        ffi12='BusEq'
    elif (4800<=row<=4899):
        ffi12='Telcm'
    elif (4900<=row<=4949):
        ffi12='Utils'
    elif (5000<=row<=5999) or (7200<=row<=7299) or (7600<=row<=7699):
        ffi12='Shops'
    elif (2830<=row<=2839) or (row==3693) or (3840<=row<=3859) or (8000<=row<=8099):
        ffi12='Hlth'
    elif (6190<=row<=6999):
        ffi12='Fin'
    else:
        ffi12='Other'
    return(ffi12)


# In[4]:


#create a pivot table with only returns and company code
table1=pd.pivot_table(data,'ret','permco','date')
print(table1)


# In[5]:


#change rows from returns to indexed prices
for i in data.permco.unique():
    arr = table1.loc[i]+1
    table1.loc[i]=arr.cumprod(skipna=True)*100
print(table1)


# In[23]:


#Graph price function
def graph_price(a,b):
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    result = pd.merge(X, Y, on='date')
    new_x = dates.datestr2num(result['date'])
    get_ipython().run_line_magic('matplotlib', 'inline')
    # Control the default size of figures in this Jupyter notebook
    get_ipython().run_line_magic('pylab', 'inline')
    pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
    ax=result["prc_x"].plot(grid = True) 
    ax=result["prc_y"].plot(grid = True) 
    ax.set_xticklabels(result['date'])
    ax.set_title('Price')
#Graph calculated price function (start with 100) change this!!
def graph_calculated_price(a,b):
    X=data.loc[data['permco']==a]
    Y=data.loc[data['permco']==b]
    result = pd.merge(X, Y, on='date')
    x1=(result['ret_x']+1).cumprod(skipna=True)*100 #closing price
    y1=(result['ret_y']+1).cumprod(skipna=True)*100 #closing price
    new_x = dates.datestr2num(result['date'])
    get_ipython().run_line_magic('matplotlib', 'inline')
    # Control the default size of figures in this Jupyter notebook
    get_ipython().run_line_magic('pylab', 'inline')
    pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
    ax=x1.plot(grid = True) 
    ax=y1.plot(grid = True) 
    ax.set_xticklabels(result['date'])
    ax.set_title('Caculated Price')


# In[108]:


graph_calculated_price(7992,8879)


# In[103]:


print(data.loc['permco'==7992])


# In[7]:


#on prices
prices=data.prc
print("PRICE standard deviation: ",np.std(prices))
print("PRICE mean: ",np.mean(prices))
print("PRICE max: ",np.max(prices))
print("PRICE min: ",np.min(prices))


# In[8]:


#on returns
returns = data.ret
print("RETURN standard deviation: ",np.std(returns))
print("RETURN mean: ",np.mean(returns))
print("RETURN max: ",np.max(returns))
print("RETURN min: ",np.min(returns))


# In[6]:


#create a list to hold industry names
diff_companies=data.permco.unique()
n=len(diff_companies)
industry_list=[]
for i in range(n):
    X=data.loc[data['permco']==diff_companies[i]]
    x1=X['siccd'].iloc[0]
    #print(x1,ffi12(x1))
    industry_list.append(ffi12(x1))
#print(diff_companies)
#print(industry_list)
#4536 different companies in total


# In[7]:


NoDur = [i for i, x in enumerate(industry_list) if x == 'NoDur']
Durbl = [i for i, x in enumerate(industry_list) if x == 'Durbl']
Manfu = [i for i, x in enumerate(industry_list) if x == 'Manfu']
Enrgy = [i for i, x in enumerate(industry_list) if x == 'Enrgy']
Chems = [i for i, x in enumerate(industry_list) if x == 'Chems']
BusEq = [i for i, x in enumerate(industry_list) if x == 'BusEq']
Telcm = [i for i, x in enumerate(industry_list) if x == 'Telcm']
Utils = [i for i, x in enumerate(industry_list) if x == 'Utils']
Shops = [i for i, x in enumerate(industry_list) if x == 'Shops']
Hlth = [i for i, x in enumerate(industry_list) if x == 'Hlth']
Fin = [i for i, x in enumerate(industry_list) if x == 'Fin']
Other = [i for i, x in enumerate(industry_list) if x == 'Other']


# In[8]:


data['date2'] = pd.to_datetime(data['date'])
#Engle-Granger Test
def EG_coin_result(a,b):  
    X=table1.loc[a][:2000]
    Y=table1.loc[b][:2000]
    #the data is from 2005 to 2015, 2517 days, define the first 2500 days as training period
    #start = datetime.datetime(2005, 1, 1)
    #end = datetime.datetime(2009, 12, 31)
    #mask = (data['date2'] > start) & (data['date2'] <= end)
    #X=X.loc[mask]
    #Y=Y.loc[mask]
    #result = pd.merge(X, Y, on='date2')
    #x1=result['prc_x'] #closing price
    #y1=result['prc_y'] #closing price
    '''
    #calculate price with 100
    newprc_x=pd.Series()
    newprc_x.set_value(result['date2'][0],100)
    for i in range(1,x1.size):
        newprc_x.set_value(result['date2'][i],newprc_x[i-1]*(result['ret_x'][i-1]+1))
    newprc_y=pd.Series(index=result['date2'])
    newprc_y.set_value(result['date2'][0],100)
    for i in range(1,y1.size):
        newprc_y.set_value(result['date2'][i],newprc_y[i-1]*(result['ret_y'][i-1]+1))
    '''
    #print ("Test 1: Engle Granger")
    try:
        coin_result = ts.coint(X, Y)
        #print("p-value: ",coin_result[1])
        #print('pair: ',a,'&',b)
        if (coin_result[1]<0.05 and not coin_result[1]==0):
            #print("passed")
            passed=True
            print("p value:",coin_result[1])
        else:
            #print("not passed")
            passed=False
    except:
        passed=False
    return passed


# In[19]:


import time
start_time = time.time()
print(7953,7954)
print("--- %s seconds ---" % (time.time() - start_time))
print(data.permco.unique().size)
#4356 choose 2 = 9485190
#9485190*0.0003180503845214844=3016 seconds = 50 minutes
#50 minutes / 10 should be 5 minutes!


# In[9]:


#run Engle-Granger Test and find pairs
unique = data.permco.unique()
n=len(NoDur)
pairs=[]
print("Industry: NoDur")
pair_count=0
for i in range(n):
    for j in range(i+1,n):
        a=unique[i]
        b=unique[j]
        if EG_coin_result(a,b):
            print("pair: ",a,b)
            pairs.append([a,b])
            pair_count=pair_count+1
possible_pairs=n*(n-1)/2
print("Finished! Found",pair_count,"pairs out of ",possible_pairs," possible pairs.")


# In[ ]:


#Therefore, the percentage of valid pairs is 3.09% within the Non Durables industry


# In[10]:


def johansen_coin_result(a,b):
    X=table1.loc[a][:2000]
    Y=table1.loc[b][:2000]
    '''
    result = pd.merge(X, Y, on='date2')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    #calculate price with 100
    newprc_x=pd.Series()
    newprc_x.set_value(result['date2'][0],100)
    for i in range(1,x1.size):
        newprc_x.set_value(result['date2'][i],newprc_x[i-1]*(result['ret_x'][i-1]+1))
    newprc_y=pd.Series(index=result['date2'])
    newprc_y.set_value(result['date2'][0],100)
    for i in range(1,y1.size):
        newprc_y.set_value(result['date2'][i],newprc_y[i-1]*(result['ret_y'][i-1]+1))
    '''
    try:
        #df=pd.DataFrame(X,Y)
        arr=np.stack((X,Y),axis=1)
        coin_result =coint_johansen(arr,0,1)
        #print ("Test 2: Johansen")
        #print("\ntest statistics: r<=0, Crit-90%, Crit-95%, Crit-99%")
        trstat = coin_result.lr1                       # trace statistic
        tsignf = coin_result.cvt                       # critical values
        #print("r<=0", trstat[0], tsignf[0])
        #print("r<=1", trstat[1], tsignf[1])
        #print('pair: ',a,'&',b)
        if (trstat[0]>tsignf[0][1]):
            #print("passed")
            passed=True
        else:
            #print("not passed")
            passed=False
        return passed
    except:
        return False


# In[11]:


#run johansen on existing pairs which passed Engle-Granger
deleted=0
pairs2=pairs
for i in pairs:
    a=i[0]
    b=i[1]
    if johansen_coin_result(a,b)==True:
        continue
    else:
        print("pair: ",a,b," didn't pass!")
        #graph_price(a,b)
        pairs2.remove([a,b])
        deleted=deleted+1
print("Finished! Deleted",deleted,"pairs!")


# In[56]:


#Among 292 pairs that passed Engle-Granger
#43 pairs didn't pass Johansen
#pass rate: 85.27%
#Now we have 249 pairs


# In[31]:


#some examples of pairs that passed Engle-Granger but didn't pass johansen
graph_calculated_price(8220,8598)


# In[12]:


#Ad-Fuller-Test on the ratio
def adf_test(a,b):
    X=table1.loc[a][:2000]
    Y=table1.loc[b][:2000]
    r1=X/Y
    # perform Augmented Dickey Fuller test
    #print('Results of Augmented Dickey-Fuller test:')
    dftest = ts.adfuller(r1.values, autolag='AIC' )
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    #print(dfoutput)
    #print(dftest[4])
    if (dftest[0]<dftest[4]['5%']):
        #print(a,b,'passed')
        passed=True
    else:
        #print(a,'Not passed')
        passed=False
    return passed


# In[13]:


#run ADF on existing pairs
deleted=0
pairs3=pairs2
for i in pairs3:
    a=i[0]
    b=i[1]
    #print(a,b,adf_test(a,b))
    if adf_test(a,b)==True:
        continue
    else:
        print("pair: ",a,b, "didn't pass!")
        pairs3.remove([a,b])
        deleted=deleted+1
print("Finished! Deleted",deleted,"pairs!")


# In[59]:


#Among 249 pairs that passed Engle-Granger and Johansen
#76 pairs didn't pass ADF
#pass rate: 69.5%
#the pass rate is much lower than last time, because Engle-Granger and Johansen are not independent
#therefore, pairs that passed one test is most likely to pass another
#Now we have 173 pairs
print(len(pairs3))


# In[34]:


#Trading Strategy: 
#Open a position when prices diverge by two Standard Deviations
#High price: short; Low price: long
#Close position when prices meet again
#Implement stop loss - 15%!
def trade_stdev(a,b):
    x1=table1.loc[a][2001:2517]
    y1=table1.loc[b][2001:2517]
    '''
    result = pd.merge(X, Y, on='date')
    result.set_index('date')
    x1=result['prc_x'] #closing price
    y1=result['prc_y'] #closing price
    '''
    diff = x1-y1
    stdeviation=np.std(diff)
    #print(stdeviation)
    #interest=0.02 #how to count the interest for shorting? assumptions at the end, will be easy
    #allocate the capital, evenly, according to the , less important
    #trading 
    profit=0
    invested=1000
    if x1.shape[0]<y1.shape[0]:
        min_shape=x1.shape[0]
    else:
        min_shape=y1.shape[0]
    trade=False
    stopped=False
    stopped_times=0
    for i in range(0,min_shape-1):
        if trade:
            #implement stop loss and close trade
            shorted_prc_now=short[i]
            longed_prc_now=long[i]
            profit_now=short_share*(short_prc-shorted_prc_now)+long_share*(longed_prc_now-long_prc)
            if profit_now<-20:
                #print("stopped")
                profit=profit_now
                trade=False
                stopped=True
                stopped_times=stopped_times+1
        if abs(x1[i]-y1[i])>stdeviation*2 and not trade:
            trade=True
            if x1[i]>y1[i]:
                long_share=1000/y1[i]
                long_prc=y1[i]
                long=y1
                short_share=(long_share*long_prc)/x1[i]
                short_prc=x1[i]
                short=x1
                print("long", long,"short",short)
                print(i,"open")
            else:
                long_share=1000/x1[i]
                long_prc=x1[i]
                long=x1
                short_share=(long_share*long_prc)/y1[i]
                short_prc=y1[i]
                short=y1
                print("long", long,"short",short)
                print(i,"open")
            #print('short',short_prc)
            #print('long',long_prc)
        if x1[i]==y1[i] and trade:
            trade=False
            close=x1[i]
            #print(close)
            invested=short_share*(short_prc-close)+long_share*close #(1-interest)
            profit=short_share*(short_prc-close)+long_share*(close-long_prc) #(1-interest)
            #print('invested',invested)
            #print('profit',profit)
            print(profit,"close")
        if trade and i==min_shape-2:
            shorted_prc_now=short[i]
            longed_prc_now=long[i]
            #print('short',shorted_prc_now)
            #print('long',longed_prc_now)
            profit=short_share*(short_prc-shorted_prc_now)+long_share*(longed_prc_now-long_prc)
            #print(profit)
            print(profit,"close")
    return_stock=profit/1000
    #print('pair:',a,b,'; return',return_stock)
    return return_stock,stopped, stopped_times


# In[76]:


trade_stdev(7976,8786)


# In[32]:


returns=[]
positive=0
max_times=0
max_pair=[]
for i in pairs3:
    a=i[0]
    b=i[1]
    #profit, stopped=trade_stdev(a,b)
    stopped_times=trade_stdev(a,b)[2]
    #if profit>0:
    #    positive=positive+1
    #else:
    #    print(a,b,stopped,profit)
    if (stopped_times>max_times):
        max_times=stopped_times
        max_pair=(a,b)
    #returns.append(profit)
print(max_times,max_pair)


# In[33]:


graph_calculated_price(8431, 8702)
print(trade_stdev(8431, 8702)[0])


# In[35]:


trade_stdev(8431, 8702)


# In[112]:


graph_price(8179,8217)


# In[ ]:





# In[19]:


#stats
print("trading return standard deviation: ",np.nanstd(returns))
print("trading return mean: ",np.nanmean(returns))
print("trading return max: ",np.nanmax(returns))
print("trading return min: ",np.nanmin(returns))
print("percent of positive returns",positive/len(returns))


# In[83]:


print("Try the stop loss of 10%")
returns=[]
positive=0
for i in pairs3:
    a=i[0]
    b=i[1]
    profit=trade_stdev(a,b)
    if profit>0:
        positive=positive+1
    returns.append(profit)
print("trading return standard deviation: ",np.nanstd(returns))
print("trading return mean: ",np.nanmean(returns))
print("trading return max: ",np.nanmax(returns))
print("trading return min: ",np.nanmin(returns))
print("percent of positive returns",positive/len(returns))


# In[86]:


print("A stop loss of 10% yields higher performance than 15%, improved percent of positive returns by 0.6%")
print("Now try the stop loss of 20%")
returns=[]
positive=0
for i in pairs3:
    a=i[0]
    b=i[1]
    profit=trade_stdev(a,b)[0]
    if profit>0:
        positive=positive+1
    returns.append(profit)
print("trading return standard deviation: ",np.nanstd(returns))
print("trading return mean: ",np.nanmean(returns))
print("trading return max: ",np.nanmax(returns))
print("trading return min: ",np.nanmin(returns))
print("percent of positive returns",positive/len(returns))


# In[87]:


print("In conclusion, performance of different stop loss thresholds:")
print("10%>15%>20%")
print("However, the difference is very small")
print("Is there a theory behind this?")
print("Now I want to see the performances of pairs which were stopped during the trading period")


# In[21]:


stopped_pairs=[]
positive=0
for i in pairs3:
    a=i[0]
    b=i[1]
    profit,stopped=trade_stdev(a,b)
    if stopped==True:
        stopped_pairs.append(profit)
        if profit>0:
            positive=positive+1
print("out of 173 pairs", len(stopped_pairs), "were stopped")
print("standard deviation: ",np.nanstd(stopped_pairs))
print("mean: ",np.nanmean(stopped_pairs))
print("max: ",np.nanmax(stopped_pairs))
print("min: ",np.nanmin(stopped_pairs))
print("percent of positive returns",positive/len(stopped_pairs))


# In[16]:


print("not stopped pairs:")
for i in pairs3:
    a=i[0]
    b=i[1]
    profit=trade_stdev(a,b)[0]
    if trade_stdev(a,b)[1]==False:
        print(a,b,profit)


# In[24]:


graph_calculated_price(8158,8179)


# In[25]:


graph_calculated_price(8778,9239)


# In[ ]:


#Another trading strategy I want to implement using rolling average and Z score
def zscore(series):
    return (series - series.mean()) / np.std(series)
#Z Score (Value) = (Value — Mean) / Standard Deviation
ratios=X/Y
ratios_mavg5 = ratios.rolling(window=5,center=False).mean()
ratios_mavg60 = ratios.rolling(window=60,center=False).mean()
std_60 = ratios.rolling(window=60,center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60


# In[27]:


#run Engle-Granger Test and find pairs
unique=data.permco.unique()
n=len(Utils)
pairs_utils=[]
print("Industry: Utils")
pair_count=0
for i in range(n):
    for j in range(i+1,n):
        a=unique[i]
        b=unique[j]
        if EG_coin_result(a,b):
            print("pair: ",a,b)
            pairs.append([a,b])
            pair_count=pair_count+1
possible_pairs=n*(n-1)/2
print("Finished! Found",pair_count,"pairs out of ",possible_pairs," possible pairs.")


# In[ ]:




