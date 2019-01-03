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


#read data from csv
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


# In[6]:


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


# In[60]:


#plot calculated price of pair 714,16640
X=data.loc[data['permco']==714]
Y=data.loc[data['permco']==16640]
result = pd.merge(X, Y, on='date')
x1=(result['ret_x']+1).cumprod(skipna=True)*100 #closing price
y1=(result['ret_y']+1).cumprod(skipna=True)*100 #closing price
new_x = dates.datestr2num(result['date'])
get_ipython().run_line_magic('matplotlib', 'inline')
# Control the default size of figures in this Jupyter notebook
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
ax=x1[2016:].plot(grid = True) 
ax=y1[2016:].plot(grid = True) 
ax.set_xticklabels(result['date'])
ax.set_title('Caculated Price')


# In[57]:


#plot z-score of pair 714 16640 (z score is: price-mean/stdev)
#Here I used code from https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a
X1=data.loc[data['permco']==714]
Y1=data.loc[data['permco']==16640]
result=pd.merge(X1,Y1,on='date')
X=result["prc_x"][2016:]
Y=result['prc_y'][2016:]
ratios=Y/X
def zscore(series):
    return (series - series.mean()) / np.std(series)
zscore(ratios).plot()
plt.axhline(zscore(ratios).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.show()


# In[61]:


#plot the 5-day and 60-day moving averages, Here I used the code from https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a
train = ratios[:2016]
ratios_mavg5 = train.rolling(window=5,center=False).mean()
ratios_mavg60 = train.rolling(window=60,center=False).mean()
std_60 = train.rolling(window=60,center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
plt.figure(figsize=(15,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
plt.ylabel('Ratio')
plt.show()


# In[65]:


# Plot the ratios and buy and sell signals from z score
plt.figure(figsize=(15,7))
train[60:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5>-1] = 0
sell[zscore_60_5<1] = 0
buy[60:].plot(color='g', linestyle='None', marker='^')
sell[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()


# In[7]:


#basic statistics on prices
prices=data.prc
print("PRICE standard deviation: ",np.std(prices))
print("PRICE mean: ",np.mean(prices))
print("PRICE max: ",np.max(prices))
print("PRICE min: ",np.min(prices))


# In[8]:


#basic statistics on returns
returns = data.ret
print("RETURN standard deviation: ",np.std(returns))
print("RETURN mean: ",np.mean(returns))
print("RETURN max: ",np.max(returns))
print("RETURN min: ",np.min(returns))


# In[9]:


#create a list to hold industry names
diff_companies=data.permco.unique()
n=len(diff_companies)
industry_list=[]
for i in range(n):
    X=data.loc[data['permco']==diff_companies[i]]
    x1=X['siccd'].iloc[0]
    industry_list.append(ffi12(x1))
#4536 different companies in total


# In[10]:


#create distinct lists of companies according to industry
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


# In[42]:


data['date2'] = pd.to_datetime(data['date'])
#Engle-Granger Test
def EG_coin_result(a,b):  
    X=table1.loc[a][:2016]
    Y=table1.loc[b][:2016]
    #the data is from 2005 to 2015, 2517 days, there are 252 trading days in a year,
    #define the first 8 years as training period, 2016 days
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
            #print("p value:",coin_result[1])
        else:
            #print("not passed")
            passed=False
    except:
        passed=False
    return passed


# In[41]:


#the pair 714,16640 passed the EG test
EG_coin_result(714, 16640)


# In[159]:


#This is a function that goes through a list and find pairs by running EG test on every possible pair
def form_pairs(ind):
    #run Engle-Granger Test and find pairs
    n=len(ind)
    pairs=[]
    pair_count=0
    for i in range(n):
        for j in range(i+1,n):
            a=diff_companies[ind[i]]
            b=diff_companies[ind[j]]
            if EG_coin_result(a,b):
                pairs.append([a,b])
                pair_count=pair_count+1
    possible_pairs=n*(n-1)/2
    print(n," companies, after EG ",pair_count,"pairs out of ",possible_pairs," possible pairs.")
    #run johansen on existing pairs which passed Engle-Granger
    deleted=0
    for i in pairs:
        a=i[0]
        b=i[1]
        if johansen_coin_result(a,b)==True:
            continue
        else:
            #graph_price(a,b)
            pairs.remove([a,b])
            deleted=deleted+1
    pair_count=pair_count-deleted
    print("after Johansen",pair_count,"pairs!")
    #run ADF on existing pairs
    deleted=0
    for i in pairs:
        a=i[0]
        b=i[1]
        #print(a,b,adf_test(a,b))
        if adf_test(a,b)==True:
            continue
        else:
            pairs.remove([a,b])
            deleted=deleted+1
    pair_count=pair_count-deleted
    print("after ADF",pair_count,"pairs!")
    return pairs


# In[157]:


#form pairs using the above function on every indutry (this can take a while)
NoDur_pairs = form_pairs(NoDur)


# In[160]:


Durbl_pairs = form_pairs(Durbl)
Manfu_pairs = form_pairs(Manfu)
Enrgy_pairs = form_pairs(Enrgy)
Chems_pairs = form_pairs(Chems)


# In[176]:


#take one industry for example
n=len(BusEq)
print(n)
BusEq_pairs=[]
pair_count=0
for i in range(n):
    for j in range(i+1,n):
        a=diff_companies[BusEq[i]]
        b=diff_companies[BusEq[j]]
        if EG_coin_result(a,b):
            print(a,b)
            BusEq_pairs.append([a,b])
            pair_count=pair_count+1
possible_pairs=n*(n-1)/2
print(n," companies, after EG ",pair_count,"pairs out of ",possible_pairs," possible pairs.")


# In[177]:


#run johansen on existing pairs which passed Engle-Granger
deleted=0
for i in BusEq_pairs:
    a=i[0]
    b=i[1]
    if johansen_coin_result(a,b)==True:
        continue
    else:
        #graph_price(a,b)
        BusEq_pairs.remove([a,b])
        print(a,b,"didnt pass!")
        deleted=deleted+1
pair_count=pair_count-deleted
print("after Johansen",pair_count,"pairs!")
#run ADF on existing pairs
deleted=0
for i in BusEq_pairs:
    a=i[0]
    b=i[1]
    #print(a,b,adf_test(a,b))
    if adf_test(a,b)==True:
        continue
    else:
        BusEq_pairs.remove([a,b])
        print(a,b,"didnt pass!")
        deleted=deleted+1
pair_count=pair_count-deleted
print("after ADF",pair_count,"pairs!")


# In[178]:


Telcm_pairs = form_pairs(Telcm)


# In[179]:


Utils_pairs = form_pairs(Utils)
Shops_pairs = form_pairs(Shops)
Hlth_pairs = form_pairs(Hlth)
Fin_pairs = form_pairs(Fin)
Other_pairs = form_pairs(Other)


# In[47]:


#johansen cointegration test function
def johansen_coin_result(a,b):
    X=table1.loc[a][:2016]
    Y=table1.loc[b][:2016]
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


# In[46]:


#our example pair, comnco 714 and 16640 passed the johansen test
johansen_coin_result(714, 16640)


# In[158]:


#Ad-Fuller-Test on the ratio
def adf_test(a,b):
    X=table1.loc[a][:2016]
    Y=table1.loc[b][:2016]
    r1=X/Y
    # perform Augmented Dickey Fuller test
    #print('Test 3: Augmented Dickey-Fuller test:')
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


# In[49]:


#714, 16640 pair passed the adf test
adf_test(714, 16640)


# In[38]:


graph_calculated_price(714, 16640)


# In[199]:


#a simple function that finds (price-mean)/stdev quicky
def zscore(trading_series,train_series):
    return (trading_series - train_series.mean()) / np.std(train_series)
#Trading Strategy: 
#Open a position when prices diverge by two Standard Deviations
#High price: short; Low price: long
#Close position when prices meet again
#Implement stop loss - 20%!
def trade_stdev(a,b):
    X=table1.loc[a]
    Y=table1.loc[b]
    x1=X[2016:]
    y1=Y[2016:]
    ratios = X/Y
    avg=ratios.rolling(window=60).mean()
    #print(len(avg))
    std = ratios.rolling(window=60).std()
    #print(len(std))
    z= (ratios[2016:] - avg[2016:])/std[2016:]
    profit=0
    invested=1000
    if x1.shape[0]<y1.shape[0]:
        min_shape=x1.shape[0]
    else:
        min_shape=y1.shape[0]
    #print("day     action   long   long price      short   short price     profit   total profit")     #table column headings
    #print("---     ------   ----   ----------      -----   -----------     ------   ------------")
    trade=False
    stopped=False
    stopped_times=0
    profits=[]
    holding_period=0
    trades=0
    for i in range(0,min_shape):
        if trade:
            #implement stop loss and close trade
            shorted_prc_now=short[i]
            longed_prc_now=long[i]
            profit_now=short_share*(short_prc-shorted_prc_now)+long_share*(longed_prc_now-long_prc)
            if profit_now<-100:
                profit=profit+profit_now
                #print(i, '\t', "stop","\t","","%.2f" % longed_prc_now,"\t","","\t","%.2f" % shorted_prc_now,"\t","%.2f" % profit_now,"\t",profit)
                profits.append(profit_now)
                trade=False
                stopped=True
                stopped_times=stopped_times+1
        if (abs(z[i])>1) and (not trade):
            trade=True
            if x1[i]>y1[i]:
                long_share=500/y1[i]
                long_prc=y1[i]
                long=y1
                short_share=500/x1[i]
                short_prc=x1[i]
                short=x1
                #print(i, '\t', "open","\t",b,"\t","%.2f" % long_prc,"\t",a,"\t","%.2f" % short_prc,"\t","","\t","%.2f" % profit)
            else:
                long_share=500/x1[i]
                long_prc=x1[i]
                long=x1
                short_share=500/y1[i]
                short_prc=y1[i]
                short=y1
                #print(i, '\t', "open","\t",a,"\t","%.2f" % long_prc,"\t",b,"\t","%.2f" % short_prc,"\t","","\t","%.2f" % profit)
            open_day=i
        if (abs(z[i])<0.75) and trade:
            trade=False
            shorted_prc_now=short[i]
            longed_prc_now=long[i]
            profit_now=short_share*(short_prc-shorted_prc_now)+long_share*(longed_prc_now-long_prc) 
            profit=profit+profit_now
            profits.append(profit_now)
            #print(i, '\t', "close","\t","","\t","%.2f" % longed_prc_now,"\t","","\t","%.2f" % shorted_prc_now,"\t","%.2f" % profit_now,"\t","%.2f" % profit)
            holding_period=holding_period+(i-open_day)
            trades=trades+1
        if trade and (i==min_shape-2):
            shorted_prc_now=short[i]
            longed_prc_now=long[i]
            profit_now=short_share*(short_prc-shorted_prc_now)+long_share*(longed_prc_now-long_prc)
            profit=profit+profit_now
            profits.append(profit_now)
            #print(i, '\t', "close","\t","","\t","%.2f" % longed_prc_now,"\t","","\t","%.2f" % shorted_prc_now,"\t","%.2f" % profit_now,"\t","%.2f" %profit)
            holding_period=holding_period+(i-open_day)
            trades=trades+1
    #print(profits)
    if(trades!=0):
        avg_holding=holding_period/trades
    else:
        avg_holding=0
    #print(avg_holding)
    return_stock=profit/2000
    return return_stock,avg_holding


# In[173]:


#trade the pair 714,16640, we can enable print commands here to see the process
trade_stdev(714, 16640)


# In[197]:


#a function that uses the trade function on every pair in a list
def trade(ind_pairs):
    returns=[]
    positive=0
    holding=0
    for i in ind_pairs:
        result=trade_stdev(i[0],i[1])
        #print("return","\t",result[0],"\t","holding","\t",result[1])
        returns.append(result[0])
        if (result[0]>0):
            positive=positive+1
        holding=holding+result[1]
    if (len(ind_pairs)!=0):
        holding=holding/len(ind_pairs)
    else:
        holding=0
    print("average return", np.nanmean(returns),"stdev",np.nanstd(returns),"percentage",positive/len(ind_pairs),"holding",holding)


# In[194]:


#trade the pairs in every industry
trade(NoDur_pairs)


# In[200]:


trade(Durbl_pairs)
trade(Manfu_pairs)
trade(Enrgy_pairs)
trade(Chems_pairs)
trade(BusEq_pairs)
trade(Telcm_pairs)
trade(Utils_pairs)
trade(Shops_pairs)
trade(Hlth_pairs)
trade(Fin_pairs)
trade(Other_pairs)


# In[ ]:




