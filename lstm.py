#%%
%load_ext autoreload
%autoreload 2

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy import array

from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #for mac os x weird bug
 

# %%

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y, direction = list(), list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)

		trend=[0,0]
		#get direction of the sequence
		if y[-1]>X[-1][-1]:
			trend[0]=1.
		else:
			trend[1]=1.
		direction.append(trend)

	return array(X), array(y), array(direction)

# %%
df=pd.read_csv(r"/Users/tom/Documents/GitHub/timeseries/data/data/AMZN.csv",parse_dates=["Date"],index_col="Date")
date=df.index
d_date=date-min(date)

#get the date and response into a nice format
t=[d.days for _,d in enumerate(d_date)]
y=df["Adj Close"].to_numpy()

#split into training and validation sets
train_lim=int(0.7*len(y))
t_train=t[0:train_lim]
y_train=y[0:train_lim]
t_valid=t[train_lim:]
y_valid=y[train_lim:]

n_features = 1
n_steps=80

#D is 1 in first col if goes up, 1 in second col if goes down.
X,Y,D=split_sequence(y_train,n_steps)
X = X.reshape((X.shape[0], X.shape[1], n_features))
# D = D.reshape((D.shape[0], D.shape[1], n_features))

Xv,Yv,Dv=split_sequence(y_valid,n_steps)
Xv = Xv.reshape((Xv.shape[0], Xv.shape[1], n_features))
# Dv = Dv.reshape((Dv.shape[0], Dv.shape[1], n_features))

plt.plot(X[0,:,0])

#%%
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(2,activation='softmax',use_bias=True))

#could do softmax to force sum to 1, or sigmoid where it has to learn to sum to 1 itself

opt=Adam(learning_rate=1e-4)
# model.compile(optimizer=opt, loss='mse')
bce=BinaryCrossentropy() 
model.compile(optimizer=opt, loss=bce, metrics=[tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives()]) 

# %% Fit the model to the exact value
#history=model.fit(X, Y, validation_data=(Xv,Yv), epochs=20, verbose=1)

#Fit the model to the direction
history=model.fit(X, D, validation_data=(Xv,Dv), epochs=5, verbose=1)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.yscale('log')
plt.show()

print(history.history.keys())

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# %%
Yhat=model.predict(Xv)
# %%
