
#%%

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

#%%

def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = load_data()

#%%

def downloadStockData(selected_stocks):
    data = yf.download(
        tickers = list(selected_stocks),
        period = "1y",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None
    )
    return(data)

data = downloadStockData(df['Symbol'])

#%%

dfSelected = df[df['GICS Sector'] == 'Financials']


keepCols=[col for col in data.columns if col[0] in dfSelected['Symbol'].values and col[1] == 'Close']

# data[keepCols].plot(legend=False)

corrData=data[keepCols].pct_change()

tempList = []

dates = corrData.index

n = 30
# for date in dates[n:-n]:
for i, date in enumerate(dates):
    if i > n:
        # 30 day lag
        date = date - datetime.timedelta(days=n)
        temp = corrData.iloc[i-n:i,:]
        corr = temp.corr()
        corrLong=pd.DataFrame(np.tril(corr), columns=corr.columns, index=corr.index).melt(ignore_index=False).query('value not in [0,1]')
        corrLong['lag'] = date
        tempList.append(corrLong)


corrAgg = pd.concat(tempList)

rollingCorr = (corrAgg
    .reset_index()
    .pivot_table(columns=['level_0', 'level_1','variable_0', 'variable_1'], values='value', index='lag')
)

# rollingCorr.iloc[:,1:10].plot(legend=False)

rollingCorr

rollingCorr=rollingCorr.melt(ignore_index=False)
rollingCorr['pair'] = rollingCorr['level_0'] + '-' + rollingCorr['variable_0']
rollingCorr.reset_index(inplace=True)

px.line(rollingCorr, x='lag', y='value', color='pair')


#%%


histoData=data[keepCols].pct_change()
# histoData.melt()['value'].hist(bins=1000)
# histoData[[('NDAQ', 'Close')]].hist(bins=50)

selec =('NDAQ', 'Close')
fig = px.histogram(histoData[[selec]].values)
fig

#%%

dfSelected = df[df['GICS Sector'] == 'Financials']

keepCols=[col for col in data.columns if col[0] in dfSelected['Symbol'].values and col[1] == 'Close']

# data[keepCols].plot(legend=False)

# corrData=data[keepCols].pct_change()
corrData=data[keepCols]

# corrData.loc[corrData.index[0]] = [100] * corrData.shape[1]

for col in corrData.columns:
    corrData[col] = (corrData[col] / corrData[col][0])*100

lineData=corrData.melt(ignore_index=False)
lineData.reset_index(inplace=True)
px.line(lineData, x='Date', y='value', color='variable_0').update_layout(showlegend=False)

#%%

sectorData = data.pct_change()
sectorData=sectorData.melt(ignore_index=False)
sectorData.query('variable_1 == "Close"', inplace=True)
sectorData.reset_index(inplace=True)

sectorData = pd.merge(
    left=sectorData,
    right=df[['Symbol', 'GICS Sector']],
    left_on='variable_0',
    right_on='Symbol').drop(['variable_0', 'variable_1'],axis=1)

sectorData=sectorData.groupby(['GICS Sector','Date'])['value'].agg('mean')
sectorData=pd.DataFrame(sectorData)
sectorData.reset_index(inplace=True)
sectorData=sectorData.pivot_table(index='Date', columns='GICS Sector', values='value')
sectorData.loc[sectorData.index[0]] = [100] * sectorData.shape[1]
sectorData
for i, index in enumerate(sectorData.index):
    if i == len(sectorData.index)-1:
        break
    sectorData.iloc[i+1]=sectorData.iloc[i] + (sectorData.iloc[i] * sectorData.iloc[i+1])


sectorData=sectorData.melt(ignore_index=False)
sectorData.reset_index(inplace=True)
sectorData
px.line(sectorData, x='Date', y='value', color='GICS Sector')




#%%

for col in rollingCorr['pair']:
    print(col)

def equal(pair1:str, pair2:str) -> bool:
    """[summary]

    Args:
        pair1 (str): ['XYZ-ZYX']
        pair2 (str): ['XYZ-ZYX']

    Returns:
        bool: [description]
    """

    sec1, sec2 = pair1.split('-')

#%%
# this is useless code there are no duplicates in the corr pairs already :)

allpairs=rollingCorr['pair'].unique()

pair1 = 'ZYX-XYZ'
pair2 = 'XYZ-ZYX'

sec1, sec2 = pair1.split('-')

keepList=[]
dropList=[]

for pair in allpairs:
    sec1, sec2 = pair.split('-')
    dropPair=sec2+'-'+sec1
    keepList.append(pair)
    dropList.append(dropPair)

keepList = [pair for pair in keepList if pair not in dropList]

rollingCorr[rollingCorr['pair'].isin(keepList)]

#%%


