import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import time
import os
from os import listdir
from os.path import isfile, join
np.random.seed(12345)

f_name = "C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/btc_data"

def get_data(symbol, interval, period):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    return data

valid_period = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']
period = input('For how many days do you want to extract data? \n Type 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max only\n')
while period not in valid_period:
  period = input('For how many days do you want to extract data? \n Type 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max only\n')


valid_interval = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
interval = input('What intervals do you want to use? \n Type 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo only\n')
while interval not in valid_interval:
  interval = input('What intervals do you want to use? \n Type 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo only\n')

symbol = 'BTC-USD'

print(f"Extracting Bitcoins data for the last {period} period of time, {interval} intervals and {symbol}")


while True:
    data = get_data(symbol, interval, period)
    data.index = data.index.tz_localize(None)
    data = data.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1)
    data.columns = ['Close_Price', 'Volume_BTC']
    list_of_files = [data]

    for file in [f for f in listdir(f_name) if isfile(join(f_name, f))]:
        if file.endswith('.csv'):
        # loads list of stored '.csv' files with tweeter data metrics and joins all days in a new Dataframe
            df = pd.read_csv(f'C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/btc_data/{file}', index_col=0)
            list_of_files.append(df)

            # should always be 2, because the first is the one we just created and the second is the one that was already there
            print(len(list_of_files))
            btcDF = pd.concat(list_of_files, axis=0)
            btcDF.drop_duplicates(keep='first', inplace=True)
            
            # once loaded and added to the list, delete the file
            try:
                os.remove(f'C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/btc_data/{file}')
            except:
                print("Error while deleting file : ", file)
    
    btcDF.index = pd.to_datetime(btcDF.index)
    btcDF = btcDF.sort_index()

    lngth = len(btcDF)
    filename = f"btc_{lngth}_entries"
    btcDF.to_csv(f'{f_name}/{filename}.csv', index=True)
    print("Sleeping for 900 seconds")
    time.sleep(900)