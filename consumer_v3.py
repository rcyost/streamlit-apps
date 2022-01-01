

import aiohttp
from collections import deque, defaultdict
from functools import partial
import asyncio
import streamlit as st

import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt

from scipy.stats import spearmanr
import yfinance as yf
import time

# Allows us to create graph objects for making more customized plots
import plotly.graph_objects as go
import plotly.express as px

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

WS_CONN = f"ws://localhost:8000/sample/"

async def consumer(graphs: dict, window_size: int, status, symbol, con):

    windows = defaultdict(partial(deque, [0]*window_size, maxlen=window_size))

    list_symbols = symbol.split('-')
    con = con + symbol

    async with aiohttp.ClientSession(trust_env = True) as session:
        status.subheader(f"Connecting to {con}")
        # send symbols here?
        async with session.ws_connect(con) as websocket:
            status.subheader(f"Connected to: {con}")
            async for message in websocket:
                data = message.json()

                windows[data['channel_0']].append(data["data_0"])
                windows[data['channel_1']].append(data["data_1"])
                windows[data['corr_channel']].append(data["corr_1_data"])

                graphs[list_symbols[0]].line_chart({list_symbols[0]: windows[list_symbols[0]]})
                graphs[list_symbols[1]].line_chart({list_symbols[1]: windows[list_symbols[1]]})
                graphs['corr'].line_chart({'corr_1': windows['corr_1']})


st.title('S&P 500 App')


st.markdown("""
Use the filters on the sidebar to select two securities. Once two have been selected, click the button to download the price data.
This will display an animated statistic generation.
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Sidebar - Sector selection
sorted_subsector_unique = sorted( df[df['GICS Sector'].isin(selected_sector)]['GICS Sub-Industry'].unique() )
selected_subsector = st.sidebar.multiselect('Sub-Sector', sorted_subsector_unique, sorted_subsector_unique)


# Stock selection
sorted_stock_unique = sorted( df[df['GICS Sub-Industry'].isin(selected_subsector)]['Security'] )
selected_stock = st.multiselect('Stock', sorted_stock_unique, sorted_stock_unique)


# Filtering data
df_selected_stocks = df[ (df['Security'].isin(selected_stock)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_stocks.shape[0]) + ' rows and ' + str(df_selected_stocks.shape[1]) + ' columns.')
st.dataframe(df_selected_stocks)



status = st.empty()
connect = st.checkbox("Connect to WS Server")

selected_channels = [symbol for symbol in df_selected_stocks.Symbol]

str_selected_channels = ''

for symbol in selected_channels:
    str_selected_channels = str_selected_channels + symbol + '-'
str_selected_channels = str_selected_channels[:-1]

st.text(str_selected_channels)
window_size = st.number_input("Window Size", min_value=10, max_value=100)

# add additional space for corr chart
selected_channels.append('corr')

# create space to put charts
columns = [col.empty() for col in st.columns(len(selected_channels))]


if connect:
    # get data from websocket
    asyncio.run(consumer(graphs=dict(zip(selected_channels, columns)),
                         window_size=window_size,
                         status=status,
                         symbol=str_selected_channels,
                         con=WS_CONN))
else:
    status.subheader(f"Disconnected.")
