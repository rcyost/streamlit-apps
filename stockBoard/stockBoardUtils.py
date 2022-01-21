


import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.express as px


def sectorChart(data, df):
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
    for i, index in enumerate(sectorData.index):
        if i == len(sectorData.index)-1:
            break
        sectorData.iloc[i+1]=sectorData.iloc[i] + (sectorData.iloc[i] * sectorData.iloc[i+1])


    sectorData=sectorData.melt(ignore_index=False)
    sectorData.reset_index(inplace=True)
    return(px.line(sectorData, x='Date', y='value', color='GICS Sector'))


def rollingCorr(data, keepCols, n=30):

    corrData=data[keepCols].pct_change()

    tempList = []

    dates = corrData.index

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

    return(px.line(rollingCorr, x='lag', y='value', color='pair'))



def lineChart(data, keepCols):
    lineData=data[keepCols]
    for col in lineData.columns:
        lineData[col] = (lineData[col] / lineData[col][0])*100

    lineData=lineData.melt(ignore_index=False)
    lineData.reset_index(inplace=True)
    return(px.line(lineData, x='Date', y='value', color='variable_0').update_layout(showlegend=False))

