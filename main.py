import pandas as pd
import numpy as np
import streamlit as st

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


def main():
    all_transactions, transactions, store_ids = get_transactions_by_store()

    st.write("## Sales Outlets")
    display_map(store_ids)

    columns = st.columns(len(store_ids))
    forecasts = []

    for store_id, col in zip(store_ids, columns):
        with col:
            store_transactions = transactions.get_group(store_id)

            st.write(f"## Forecast for store #{store_id}")
            forecast = predict_future_sales(store_transactions)
            forecasts.append(forecast)

    st.write(f"## Forecast for all store")
    predict_future_sales(all_transactions, 100)


def display_map(store_ids):
    df = pd.read_csv('./datasets/sales_outlet.csv')
    df = df.rename(columns={'store_longitude': 'lon', 'store_latitude': 'lat'})
    st.map(df)


def get_transactions_by_store():
    st.write(" # Loading transactions data from local dataset ")
    df = pd.read_csv('./datasets/transactions.csv')
    df = df.rename(columns={'transaction_date': 'ds', 'line_item_amount': 'y'})
    df = df[df.instore_yn != 'N']
    df = df[df.y < 60]
    df['cap'] = 60
    df['floor'] = 0
    st.write(df.head())

    txs = df.groupby(['sales_outlet_id'])
    return df, txs, txs.groups


def predict_future_sales(store_transactions, periods=30):
    model = Prophet(growth='logistic')
    model.fit(store_transactions)
    future = model.make_future_dataframe(periods)
    future['cap'] = 60
    future['floor'] = 0

    st.write("Use model to predict in future dataframe")
    forecast = model.predict(future)

    st.write(forecast.head())

    st.write(f"## Forcast ")
    st.write(model.plot(forecast))
    st.write(model.plot_components(forecast))
    return forecast


if __name__ == '__main__':
    main()
