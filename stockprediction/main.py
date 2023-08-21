import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'BABA', 'PYPL', 'ADBE', 'INTC', 'TWTR', 'CRM', 'IBM', 'DIS' )
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
forecast_period = n_years * 252  # Assuming 252 trading days in a year

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Raw Data')
    plt.legend()
    st.pyplot()

plot_raw_data()

# Prepare data for forecasting
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = pd.to_datetime(df_train['ds'])

# Forecast with Exponential Smoothing
model = ExponentialSmoothing(df_train['y'], trend="add", seasonal="add", seasonal_periods=30)
model_fit = model.fit()

future_dates = pd.date_range(start=df_train['ds'].iloc[-1], periods=forecast_period, freq='B')
forecast = model_fit.forecast(steps=forecast_period)
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
st.write("Forecast components")

# Extract and display actual, predicted, and residual values
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(df_train['ds'], df_train['y'], label='Actual')
ax1.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', linestyle='dashed')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.set_title('Forecast and Actual')
ax1.legend()

# Display the figure using st.pyplot()
st.pyplot(fig1)

# Calculate residuals
forecast_df['Residuals'] = df_train['y'].iloc[-1] - forecast_df['Forecast']

# Plot residuals
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(forecast_df['Date'], forecast_df['Residuals'], label='Residuals')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual')
ax2.set_title('Residuals')
ax2.legend()

# Display the figure using st.pyplot()
st.pyplot(fig2)
st.set_option('deprecation.showPyplotGlobalUse', False)