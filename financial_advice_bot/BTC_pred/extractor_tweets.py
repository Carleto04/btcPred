## import libraries
#--------------------------------------------------------------------#
import pandas as pd
import numpy as np
import datetime
import tweepy
import os
from datetime import date
import time
from random import randint
from time import sleep
from os import listdir
from os.path import isfile, join
from authpy import authpy
np.random.seed(12345)

f_name = "C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/tweets_per_days"

api = authpy()

# Authenticate to Twitter verification
try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')


## Get tweets that contain the hashtag #btc or #bitcoin
## "-is:retweet" means I don't want retweets
## "lang:en" is asking for the tweets to be in english
query = '#btc OR #bitcoin -is:retweet lang:en'
today = datetime.date.today()
limit = 1000

print(f"Extracting Twitter data for {limit} entries with the search {query}")

while True:
      
    msgs = []
    not_in_msgs = []
    msg =[]

    try:
        for tweet in tweepy.Cursor(api.search_tweets, q=query).items(limit):
            msg = [tweet.text, tweet.created_at]
            msg = tuple(msg)
            # if the tweet is not duplicated, add it to the list
            if msg not in msgs:
                msgs.append(msg)
            else:
                not_in_msgs.append(msg)
    except tweepy.TweepyException:
        print("Waiting for 900 seconds due to Exceeded limit error")
        time.sleep(15 * 60)


    # verify how long is the data extracted
    lngth = len(msgs) # stored in a different variable because we will use this for the name of the file at the end
    print(f'{lngth} good data')
    print(f'{len(not_in_msgs)} duplicated data')

    # convert values from API to usable Dataframe
    Vnotclean = pd.DataFrame(msgs, columns=['text', 'date'])

    Vnotclean.date = Vnotclean['date'].dt.tz_localize(None)

    # verify resulting Dataframe
    print(Vnotclean.info())
    print(Vnotclean)

    # this code checks if there is files ending with '.csv' in the folder an lists them all in 'list_files'
    list_files = [Vnotclean]
    for file in [f for f in listdir(f_name) if isfile(join(f_name, f))]:
        if file.endswith('.csv'):
        # loads list of stored '.csv' files with tweeter data metrics and joins all days in a new Dataframe
            df = pd.read_csv(f'{f_name}/{file}')
            list_files.append(df)
            # should always be 2
            print(len(list_files))

            # merge all Twitter Dataframes and drop duplicate values
            Vnotclean = pd.concat(list_files, axis=0)
            Vnotclean.drop_duplicates(keep='first', inplace=True)
            Vnotclean.head()

            # once loaded, added to the list and merged, delete the past file
            try:
                os.remove(f'{f_name}/{file}')
            except:
                print("Error while deleting file : ", file)


    # Save Dataframe if necessary with the historical data in a csv file
    lngth = len(Vnotclean)
    file_name = f"to_{today}_{lngth}_tweets"
    Vnotclean.to_csv(f'{f_name}/{file_name}.csv', index=False)
    print("Sleeping for 1800 seconds")
    time.sleep(1800)