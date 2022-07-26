from os import listdir
from os.path import isfile, join
import pandas as pd
from twitt_clean_scan import cleaner
from fractioner import df_frac

# ....... DATA Twitter ....... #
## this code checks if there is files ending with '.csv' in the folder and loads it
f_name = "C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/tweets_per_days"
for file in [f for f in listdir(f_name) if isfile(join(f_name, f))]:
        if file.endswith('.csv'):
        # loads list of stored '.csv' files with tweeter data
            notclean = pd.read_csv(f'{f_name}/{file}')

# cleaner removes non relevant data and outputs a dataframe with polarity, sensitivity and tweet volume
twDF = cleaner(notclean)
print(twDF)

# ....... DATA BTC ....... #
## this code checks if there is files ending with '.csv' in the folder and loads it
f_name2 = "C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/btc_data"
for file in [f for f in listdir(f_name2) if isfile(join(f_name2, f))]:
        if file.endswith('.csv'):
        # loads list of stored '.csv' files with bitcoin data
            btcDF = pd.read_csv(f'{f_name2}/{file}', index_col=0)

# change the index from datatype=str to datetime
btcDF.index = pd.to_datetime(btcDF.index)

# Drop NAN values and create a new column, representing the difference between each hour close price and the previous
btcDF = btcDF.dropna()
btcDF['Diff'] = btcDF.Close_Price.pct_change() * 100
btcDF = btcDF.drop(columns='Close_Price')
print(btcDF)

print(btcDF.index.dtype)
print(twDF.index.dtype)

# ....... joined data ....... #
# merge BTC and Tweeter data in a new Dataframe.
## inner: use intersection of keys from both frames and preserve the order of the left keys.
## left_index and right_index use the index from the left and right DataFrame as the join key(s),
Final_df = pd.merge(twDF,btcDF, how='inner',left_index=True, right_index=True)

# Use the fractioner if the resulting dataframe is too big
Final_df = df_frac(Final_df)


print(Final_df.info())
print(Final_df)
print(Final_df.isna().sum())
print(len(Final_df))

# Save Dataframe as .csv file if necessary
f_name3 = "C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data"
Final_df.to_csv(f'{f_name3}/tw_btc_data.csv', index=True)