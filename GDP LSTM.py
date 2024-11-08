import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
f = pd.read_csv("C:/Users/srafi/Desktop/Aman Khan Stuffs/Old work/Data.csv")
GDP = f['Real GDP']
series = GDP.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series.reshape(-1, 1))
train_size = int(len(scaled_series)*0.67)
test_size = len(scaled_series)- train_size
train, test  = scaled_series[0:train_size,:], scaled_series[train_size:len(scaled_series),:]
def create_dataset(scaled_series, look_back=1):
	dataX, dataY = [], []
	for i in range(len(scaled_series)-look_back-1):
		a = scaled_series[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(scaled_series[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
    # reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=79, batch_size=1, verbose=2)
model.summary()
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(scaled_series)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(scaled_series)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(scaled_series)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(scaled_series),label="GDP")
plt.plot(trainPredictPlot,label="Train GDP")
plt.plot(testPredictPlot,label="Test GDP")
plt.legend(loc="upper left")
plt.show()
plt.show()
