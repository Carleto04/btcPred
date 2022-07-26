import datetime
from datetime import date, timedelta
import keras
import pandas as pd
from pandas import concat
from pandas import DataFrame
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

today = datetime.date.today()
yesterday = date.today() - timedelta(days=1)

# load preset model
model_name = f"{today}_prediction_model"
model = keras.models.load_model(f'/content/drive/MyDrive/Colab Notebooks/Data/{model_name}.h5')
print(model)

# load all the previously recorded data
Final_df = pd.read_csv(r'/content/drive/MyDrive/Colab Notebooks/Data/tw_btc_data.csv', index_col=0)
Final_df.index = pd.to_datetime(Final_df.index)

# make a set of data out of the last 2 days
df_twoday = Final_df.loc[str(yesterday):str(today)]


# --------------- Here Starts the data processing --------------- #
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

values = df_twoday.values
cols = df_twoday.columns.tolist()

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


# --------------- PREDICTIONS FROM HERE --------------- #
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

# visualize the prediction
plt.plot(inv_y, label='Real')
plt.plot(inv_yhat, label='Predicted')
plt.title('Real v Predicted Close_Price')
plt.ylabel('Price ($)')
plt.xlabel('epochs (Hr)')
plt.legend()
plt.show()

# Create a dataframe with a column for predictions (pred) and another for real data (real)
data = {'real': inv_y, 'pred': inv_yhat}
df = pd.DataFrame(data)

# Difference between actual and previous values in %
df['Diff_r'] = df.real.pct_change() * 100
df['Diff_p'] = df.pred.pct_change() * 100

# define the starting price
price= df.real[1]

# the first value is the starting point, so we get rid of it for the predictions
df = df[1:]

# Define strategy depending on values. Values "Steady" (=0), "Sell" (<0), "Buy" (>0)
def decision(lst):
  direction = []
  for n in lst:
    if n == 0:
      direction.append('Steady')
    elif n < 0:
      direction.append('Sell')
    else:
      direction.append('Buy')
  return direction

# Append the strategy to the Dataframe
df['direction_r'] = decision(df['Diff_r'])
df['direction_p'] = decision(df['Diff_p'])
print(df)

# Test the strategy starting with 1 bitcoin
count = 1
benefit = 0
init_price = price
max_count=5

for i in range(1,len(df)+1):
  if df.direction_p[i] == 'Buy':
    if count < max_count:
      benefit -= price
      price = df.real[i]
      count += 1
      print(benefit, price, count)
  elif df.direction_p[i] == 'Sell':
    if count > 0:
      benefit += price
      price = df.real[i]
      count -= 1
      print(benefit, price, count)

earnings = benefit + count*price - init_price
pc_earn = earnings * 100 / init_price

# Printing the strategy results test. 
# Consider only valid the % differences, since the absolute values are scalated for predicting
print("You started at", init_price)
print("Earnings equal to", earnings, "meaning a", pc_earn, "% development")
print("You have", count, "BTCs")