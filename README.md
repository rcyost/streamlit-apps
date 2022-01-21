# streamlitPlay

## Goal: explore using streamlit to make apps in python

Seems comparable to flexboard in R though a bit more features.

Found a nice example of a realtime app and played around with it.

Didn't work with state at all so still a bit basic.

## StockBoard:

 - downloads data from yahoo finance
 - creates plotly charts

## Real Time Example

- producers creates random variables and corr between them
- sends over port using uvicorn
- consumer uses aync function to add to charts
- since doing this streamlist has changed the way to add new data to charts

to run real time app in cli type:

streamlit run consumer_v3.py

uvicorn producer_v3:app

## Rolling Corr
- practicing adding data to charts
- since doing this streamlist has changed the way to add new data to charts