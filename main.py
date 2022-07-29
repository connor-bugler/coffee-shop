import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


from datetime import datetime
from prophet import Prophet


def main():
    all_transactions, transactions, store_ids = get_transactions_by_store()
    st.write("# Coffee Shop - Sales Trends")
    st.map(store_data())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("# Store Analytics")

    with col2:
        store_id = st.selectbox('Select a store.', ('All', *store_ids))

    with col3:
        st.write("## Store Forecast")

    with col4:
        periods = st.number_input(
            'Forecast Period',
            value=30,
            min_value=10,
            max_value=100)

    if store_id == 'All':
        df = all_transactions
    else:
        df = transactions.get_group(store_id)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transactions", len(df))
        st.metric("Transactions per day",
                  math.floor(df.groupby('ds').count()['y'].mean())
                  )

    with col2:
        forecast, model = predict_future_sales(df, int(periods))
        st.write(model.plot(forecast))
        st.write(model.plot_components(forecast))

    with col1:
        st.write("## Sales by Date")
        date = st.date_input('Select Date Range.',
                             value=datetime(2019, 4, 1),
                             min_value=datetime(2019, 4, 1),
                             max_value=datetime(2019, 4, 29))
        # df = df[df.ds == date]  # .strftime('%Y-%m-%d')]
        st.write("Transactions on ", date)
        st.write(df[df.ds == str(date)])
        st.write("Cumulative sales on ", date)
        st.line_chart(df[df.ds == str(date)]
                      [['y']]
                      .rename(columns={'y': 'Net sales'})
                      .cumsum())


@st.cache
def store_data():
    return pd.read_csv('./datasets/sales_outlet.csv') \
        .rename(columns={'store_longitude': 'lon', 'store_latitude': 'lat'})


@st.cache(allow_output_mutation=True)
def get_transactions_by_store():
    df = pd.read_csv('./datasets/transactions.csv') \
        .rename(columns={'transaction_date': 'ds', 'line_item_amount': 'y'})

    df = df[df.instore_yn != 'N']
    df = df[df.y < 60]
    df['cap'] = 60
    df['floor'] = 0

    txs = df.groupby(['sales_outlet_id'])
    return df, txs, txs.groups


@st.cache
def predict_future_sales(store_transactions, periods=30):
    model = Prophet(growth='logistic')
    model.fit(store_transactions)
    future = model.make_future_dataframe(periods)
    future['cap'] = 60
    future['floor'] = 0

    return model.predict(future), model


if __name__ == '__main__':
    main()
    import sales_weather_forecast
