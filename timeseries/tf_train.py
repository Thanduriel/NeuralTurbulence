import time
import os
import shutil
import sys
import math
import random

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("../../tools")
sys.path.append("../utils")
import tilecreator
from loaddata import loadData

np.random.seed(13)
tf.random.set_seed(13)


# path to sim data, trained models and output are also saved here
basePath = '../data/'

trainingEpochs = 2500
batchSize      = 10
inSize         = 64 * 64 * 4 # warning - hard coded to scalar values 64^2
inStepsPerSim  = 256
timeFrame      = 8
seriesPerSim   = inStepsPerSim - timeFrame

# load data
inputFrames = []
outputFrames = []

# start reading simSimple 1000 ff.
for sim in range(1000,2000): 
	dir = "{}/simSimple_{:04d}/".format(basePath,sim)
	if not os.path.exists( dir ):
		break

	densities = loadData(dir + "density_*.uni")
	velocities = loadData(dir + "vel_*.uni")

	# create time series only from the same simulation
	densities = np.reshape( densities, (len(densities), 64,64,1) )
	velocities = np.reshape( velocities, (len(velocities), 64,64,3))
	data = np.concatenate((densities,velocities), axis=3)
	for i in range(0, seriesPerSim):
		input = []
		for j in range(0, timeFrame):
			input.append(data[i+j].flatten())
		inputFrames.append(input)
		outputFrames.append(data[i+timeFrame])

validationIn = inputFrames[-seriesPerSim:];
validationOut = outputFrames[-seriesPerSim:];
validationIn = np.reshape(validationIn, (len(validationIn),timeFrame,inSize))#64,64,4
validationOut = np.reshape(validationOut, (len(validationOut),inSize))

inputFrames = inputFrames[:-seriesPerSim];
outputFrames = outputFrames[:-seriesPerSim];
inputFrames = np.reshape(inputFrames, (len(inputFrames),timeFrame,inSize))
outputFrames = np.reshape(outputFrames, (len(inputFrames),inSize))

BATCH_SIZE = 16
BUFFER_SIZE = 1000

trainData = tf.data.Dataset.from_tensor_slices((inputFrames, outputFrames))
trainData = trainData.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

validData = tf.data.Dataset.from_tensor_slices((validationIn, validationOut))
validData = validData.batch(BATCH_SIZE)


# set up the network
model = keras.models.Sequential([
	layers.LSTM(64, input_shape=inputFrames.shape[-2:]),
	layers.Dense(inSize)
#	layers.Reshape((64,64,4))
])

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam())

# now we can start training...
print("Starting training...")
history = model.fit(trainData,
                    epochs=10,
                    validation_data=validData)

model.save("mymodel.h5")

