import json
from threading import active_count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import os
import random
import tensorflow as tf
import tensorflow.python as tfp
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Flatten, Input, Softmax
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as plticker

# from symbol import continue_stmt

def createModel():
    model = tfp.keras.Sequential([
        tfp.keras.layers.LSTM(4, input_shape=(3, 1)),
        # tfp.keras.layers.InputLayer(input_shape=(32, 32, 3))
        tfp.keras.layers.Dense(1)

    ])
    return model


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def process_dataset(dataset= 'dataset/dataset_sin_S10F10.csv'):
    fp = open(dataset)
    lines = fp.readlines()

    bin_lines = ['time, sensor, \n']

    for i,line in enumerate(lines):
        # input(i)
        if str(i) == '0':
            continue
        splitted = line.split(',')[1:-1]
        v = [x for x in splitted if x != ''][0]
        idx = splitted.index(v)
        bin_lines.append(f'{i-1}, {idx},\n')


    fp = open(dataset+'.bins.csv',"w+")
    fp.writelines(bin_lines)


tf.random.set_seed(7)


dataset= 'dataset/dataset_sin_S10F10.csv'
process_dataset(dataset=dataset)




dataframe = pd.read_csv(dataset+'.bins.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))




model = createModel()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)




trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
trainPredictPlot = np.rint(trainPredictPlot)

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
testPredictPlot = np.rint(testPredictPlot)

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

# loc = plticker.MultipleLocator(base=1.0)
plt.yticks(np.arange(0,20))
plt.show()

