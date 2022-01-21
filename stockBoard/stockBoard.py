import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import plotly.express as px

from scipy.stats import spearmanr
import yfinance as yf
import time

# Allows us to create graph objects for making more customized plots
import plotly.graph_objects as go
import plotly.express as px

from stockBoardUtils import *

st.set_page_config(layout="wide")
st.title('stockBoard')

st.sidebar.header('User Input Features')
st.sidebar.text('Changing period or interval downloads')
st.sidebar.text('new data')
st.sidebar.text('avoid changing too much')
st.sidebar.text('else yahoo finance might limit you')


selectedPeriod = st.sidebar.multiselect('Period', ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], ['1y'])
selectedInterval = st.sidebar.multiselect('Interval', ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], '1d')


# Web scraping of S&P 500 data
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

@st.cache
def downloadStockData():
    data = yf.download(
        tickers = list(df['Symbol']),
        period = selectedPeriod[0],
        interval = selectedInterval[0],
        group_by = 'ticker',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None
    )
    return(data)

df = load_data()
sector = df.groupby('GICS Sector')

################################################################## Sidebar
# Sidebar - Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Sidebar - Sector selection
sorted_subsector_unique = sorted( df[df['GICS Sector'].isin(selected_sector)]['GICS Sub-Industry'].unique() )
selected_subsector = st.sidebar.multiselect('Sub-Sector', sorted_subsector_unique, sorted_subsector_unique)

# Stock selection
sorted_stock_unique = sorted( df[df['GICS Sub-Industry'].isin(selected_subsector)]['Security'] )
selected_stock = st.sidebar.multiselect('Stock', sorted_stock_unique, sorted_stock_unique)



################################################################## Main Page
if len(selected_sector) != 1:
    st.stop()

# Filtering data
df_selected_stocks = df[ (df['Security'].isin(selected_stock)) ]

st.header('Display Companies in Selected Sector')
st.write('Intraday data cannot extend last 60 days')
st.write('There are: ' + str(df_selected_stocks.shape[0]) + ' stocks in the ' + str(selected_sector[0]) + ' sector')
st.dataframe(df_selected_stocks)


# https://pypi.org/project/yfinance/

data = downloadStockData()


symbols = df_selected_stocks.Symbol.unique()

keepCols=[col for col in data.columns if col[0] in symbols and col[1] == 'Close']



st.header('Charts')
st.text('=====================================================================================================================================================================================')

col1, col2 = st.columns([1, 1])

with col1:
    st.header('Sector Average Returns')
    st.plotly_chart(sectorChart(data, df))

    st.header('Selected Equities Sector Returns')

    st.plotly_chart(lineChart(data, keepCols))

    st.header('Selected Equities Rolling Correlations')
    n=st.slider(f'Window Size units: {selectedInterval[0]}', min_value=1, max_value=100)
    st.plotly_chart(rollingCorr(data, keepCols, n=n))

with col2:
    st.header(f'{selected_sector[0]} Sector {selectedInterval[0]} Return Histogram')
    histoData=data[keepCols].pct_change()
    st.plotly_chart(px.histogram(histoData.melt(), x="value"))

    st.header(f'{selected_sector[0]} Sector Individual Equity {selectedInterval[0]} Historgram')
    histoData=data[keepCols].pct_change()
    singleHistoStock = st.multiselect('Select stock for individual histogram', symbols, symbols[0])
    if len(singleHistoStock) != 1:
        st.stop()
    selec =(singleHistoStock[0], 'Close')
    st.plotly_chart(px.histogram(histoData[[selec]].values))





# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")



