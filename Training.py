import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

acc = pd.read_csv('Train.txt',sep="\n", header=None)
Fs = 1.0e6 # sampling frequency
Dt = 1.0e-6
t = np.arange(0,len(acc)*Dt,Dt)

batch_size = 5 

# add features with shifted columns
data = pd.DataFrame()
for i in range(0, batch_size):
    data[str(i)] =  acc.shift(-i).values.flatten()
    
# convert an array of values into a dataset matrix
def create_dataset(dataset, seq_len):
    dataX, dataY = [], []
    for i in range(len(dataset) - seq_len - 22):
        a = dataset[i:(i + seq_len), :]
        dataX.append(a)
        dataY.append(dataset[i + seq_len+1, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1
look_back = 10 
trainX, trainY = create_dataset(data.values, look_back)  

# reshape input to be  [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, batch_size))

tf.keras.backend.clear_session()
tf.random.set_seed(7)
np.random.seed(7)

learning_rate = 0.005 
hidden_units = 200 
nepoch = 2 

# define model
model = Sequential()
model.add(LSTM(hidden_units, input_shape=(look_back, batch_size)))
model.add(Dense(1, activation='linear'))
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
model.compile(loss="mean_squared_error", 
              optimizer=optimizer,
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(trainX, trainY, nepoch)
model.summary()

print("Layer weights {}".format(model.get_weights()))
# get the weights and bias from the tensorflow model
for i, layer in enumerate(model.get_weights()):
    tf.summary.histogram('layer{0}'.format(i), layer)

# Plots the history of model training
plt.figure()
plt.plot(history.history["loss"],label='train')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
    
# make predictions
trainPredict = model.predict(trainX)
 
# evaluate the model
train_result = model.evaluate(trainX, trainY, verbose=0)
print('train RMSE:', train_result[1])
exp_acc = acc[11:1980].values
error = np.abs(exp_acc - trainPredict)
print('naive ratio = ', np.sum(error)/np.sum(np.abs(exp_acc[0:-1]-trainPredict[1:])))

# viasualize results
plt.figure()
plt.plot(t[11:1980], acc[11:1980], label = 'experimental data')
plt.plot(t[11:1980], trainPredict, '--', label = 'prediction')
plt.xlabel('time')
plt.ylabel('g_n')
plt.legend()
plt.title('Tensorflow prediction')
