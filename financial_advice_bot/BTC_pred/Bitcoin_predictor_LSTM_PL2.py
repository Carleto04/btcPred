# ....... LSTM Model ....... #
from math import sqrt
from numpy import concatenate
from datetime import datetime
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
np.random.seed(12345)

f_name = 'C:/Users/tolay/OneDrive/Escritorio/testvisualcode/financial_advice_bot/BTC_pred/data'
Final_df = pd.read_csv(f'{f_name}/tw_btc_data.csv', index_col=0)
today = datetime.date.today()

print(Final_df)

'''
# Split data between today+yesterday and the rest of the data
## We need 2 days of data for the final step because of the needed shape for the model
## In this step of the pipeline we will use only the rest of the data
df_twoday = Final_df.loc[str(yesterday):str(today)]
Final_df = Final_df.loc[~Final_df.index.isin(df_twoday.index)]
print(Final_df)
print(df_twoday)
# Save today's data for next step in the pipeline (prediction)
df_twoday.to_csv('/content/drive/MyDrive/Colab Notebooks/Data/tw_btc_data_twodays.csv', index=True)
'''

cor = Final_df.corr()
print(cor)

sns.set(style="white")
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax =sns.heatmap(cor, cmap=cmap, vmax=1, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .7})
plt.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = DataFrame(data)
  cols, names = list(), list()
# input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

'''# print the number of train_hours (difference between the first and the last values from date)
date_time_obj = datetime.datetime.strptime(str(Final_df.index[0]), '%Y-%m-%d %H:%M:%S')
date_time_obj2 = datetime.datetime.strptime(str(Final_df.index[-1]), '%Y-%m-%d %H:%M:%S')
## Get interval between two timstamps as timedelta object
diff = date_time_obj2 - date_time_obj
## Get interval between two timstamps in hours
diff_in_hours = int(diff.total_seconds() / 3600)
print('Difference between first and last datetimes in hours:')
print(diff_in_hours)'''

values = Final_df.values
cols = Final_df.columns.tolist()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

n_hours = 3 #adding 3 hours lags creating number of observations 
n_features = 5 #Features in the dataset.
n_obs = n_hours*n_features

reframed = series_to_supervised(scaled, n_hours, 1)
reframed.drop(reframed.columns[-4], axis=1)
print(reframed.head())

values = reframed.values
n_train_hours = int(values.shape[0]*0.9)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print(train.shape)
print(test.shape)

# split into input and outputs
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=6, validation_data=(test_X, test_y), verbose=2, shuffle=False,validation_split=0.2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours* n_features,))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
mse = (mean_squared_error(inv_y, inv_yhat))
print('Test MSE: %.3f' % mse)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y, label='Real')
plt.plot(inv_yhat, label='Predicted')
plt.title('Real v Predicted Close_Price')
plt.ylabel('Price ($)')
plt.xlabel('epochs (Hr)')
plt.legend()
plt.show()

# Saving the model
model_name = f"{today}_prediction_model"
model.save(f'{f_name}/{model_name}.h5')