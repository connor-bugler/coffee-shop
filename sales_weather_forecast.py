import numpy as np
import pandas as pd
import streamlit as st

import datetime
from prophet import Prophet


def sales_weather_forecast():
    col1, col2 = st.columns(2)

    with col1:
        wf = pd.read_csv('./datasets/weather_forecast_4-6.csv')
        wf['ds'] = pd.to_datetime(wf['date_valid_std'])

        st.write("## Weather forecast data")
        st.write(wf)

    with col2:
        df = pd.read_csv('./datasets/results_10000.csv')
        df = df.rename(columns={'transaction_date': 'ds', 'sale_amount': 'y'})
        df['cap'] = 60
        df['floor'] = 0

        st.write(" ## Transaction data ")
        avg_temp = df['avg_temp'].mean()
        df['hot_day'] = df['avg_temp'].apply(
            lambda avg: 1 if avg > avg_temp else 0)

        st.write('average temp', avg_temp)
        st.write("results table", df.head())

    m = Prophet('logistic')
    m.add_regressor('hot_day')
    m.fit(df)

    def forecast_weather(ds):
        df = wf[(wf['ds'] >= ds) & (wf['ds'] < ds +
                                    datetime.timedelta(days=1))]
        avg = df.iloc[0]['avg_temp']
        return 1 if avg > avg_temp else 0

    future = m.make_future_dataframe(periods=30)
    future['ds'] = pd.to_datetime(future['ds'])
    future['cap'] = 60
    future['floor'] = 0
    future['hot_day'] = future['ds'].apply(forecast_weather)

    st.write('Predicting ...')

    forecast = m.predict(future)
    st.write(forecast.tail())

    st.write(m.plot(forecast))
    st.write(m.plot_components(forecast))
