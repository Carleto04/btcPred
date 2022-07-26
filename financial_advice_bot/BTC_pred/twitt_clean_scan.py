import pandas as pd
import re
from textblob import TextBlob

'''f_name = "C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data/tweets_per_days"
for file in [f for f in listdir(f_name) if isfile(join(f_name, f))]:
        if file.endswith('.csv'):
        # loads list of stored '.csv' files with tweeter data
            notclean = pd.read_csv(f'{f_name}/{file}')'''

def cleaner(df_notclean):
    # verify and clean nan values 
    notclean = df_notclean.dropna()

    # clean tweets text from noisy symbols
    txt = []
    clean_txt = []
    for utt in notclean.text:
        txt.append(str(utt))
    for i in range(len(txt)):
        clean_txt.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", txt[i]).split()))

    # this part of the code gets values from 0 to 1 for polarity and subjectivity using TextBlob library
    ## understand polarity as sentiment positive or negative
    ## understand sensitivity as subjectivty, how likely is the person giving an opinion
    polarity_score = []
    sensitivity_score = []
    for e in clean_txt:
        polarity_score.append(TextBlob(e).sentiment.polarity)
        sensitivity_score.append(TextBlob(e).sentiment.subjectivity)

    # add values to Tweeters Dataframe in new columns
    notclean['Polarity'] = polarity_score
    notclean['Sensitivity'] = sensitivity_score
    
    notclean['date'] = pd.to_datetime(notclean['date'])

    # use of floor mehod, which will round all the times to the hour
    notclean['DateTime'] = notclean['date'].dt.floor('h')

    # Create a new feature ‘Tweet vol’ by aggregating and grouping all the tweets by 'H'= hour
    # and create a new Dataframe called vdf
    vdf = notclean.groupby(pd.Grouper(key='DateTime',freq='H')).size().reset_index(name='tweet_vol')

    # pass dateTime values to usable format and set them to index
    vdf.index = pd.to_datetime(vdf.index)
    vdf=vdf.set_index('DateTime')

    # change type of column 'tweet_vol' from int64 to float
    vdf['tweet_vol'] =vdf['tweet_vol'].astype(float)

    # set DateTime as index
    notclean.index = pd.to_datetime(notclean.index)
    notclean =notclean.drop('text', axis=1)

    # In this step we create the last dataframe using the "mean of Polarity and Sensitivity" and "the number of Tweets per day"
    ## Use of "mean of Polarity and Sensitivity" in a single new dataframe
    df_clean = notclean.groupby('DateTime').agg(lambda x: x.mean())
    df_clean = df_clean.drop('date', axis=1)
    df = pd.merge(vdf, df_clean, left_index=True, right_index=True)
    return df