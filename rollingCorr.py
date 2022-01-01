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

# an attempt to learn how to update a chart with new data in streamlit

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


# https://pypi.org/project/yfinance/
if st.button('Once two stocks are selected click to download data'):
    if len(selected_stock) == 2:
        data = yf.download(
                tickers = list(df_selected_stocks.Symbol),
                period = "1mo",
                interval = "5m",
                group_by = 'ticker',
                auto_adjust = True,
                prepost = False,
                threads = True,
                proxy = None
            )
        st.text('Downloaded Data ok')
    else:
        st.text('Select only two stocks')



# data = pd.DataFrame({'stock1' : [np.random.randn(1)[0]],
#                       'stock2' : [np.random.randn(1)[0]] })

symbols = df_selected_stocks.Symbol.unique()
st.dataframe(data)

data = pd.DataFrame({symbols[0] : data[(symbols[0], 'Close')],
                     symbols[1] : data[(symbols[1], 'Close')]}).reset_index()

corr_data = pd.DataFrame({symbols[0] : data[symbols[0]][0],
                          symbols[1] : data[symbols[1]][0]}, index=[0])

# start the charts with first entry of data
chart0 = st.line_chart(data[symbols[0]][0:1])
chart1 = st.line_chart(data[symbols[1]][0:1])


corr = pd.DataFrame({'corr' : [np.float64(0)]})
chart2 = st.line_chart(corr['corr'])

# this loop needs to be async
for i in range(1, len(data)):

    # new_data = pd.DataFrame({'stock1' : [np.random.randn(1)[0]],
    #                          'stock2' : [np.random.randn(1)[0]] })

    new_data = pd.DataFrame({symbols[0] : data[symbols[0]][i-1:i],
                             symbols[1] : data[symbols[1]][i-1:i] })

    # df to calculate corr on
    corr_data = corr_data.append(new_data)
    # calculate step corr
    if i>30:
        new_corr = pd.DataFrame({'corr' : [spearmanr(corr_data[symbols[0]][i-30:i], corr_data[symbols[1]][i-30:i])[0]] })
    else:
        new_corr = pd.DataFrame({'corr' : [np.float64(0)]})

    corr = corr.append(new_corr)


    chart0.add_rows(new_data[symbols[0]])
    chart1.add_rows(new_data[symbols[1]])
    chart2.add_rows(new_corr['corr'])

    time.sleep(0.002)

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")



