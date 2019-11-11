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
import ioext

np.random.seed(13)
tf.random.set_seed(13)


# path to sim data, trained models and output are also saved here
basePath = '../data/'

trainingEpochs = 2500
batchSize      = 10
timeFrame      = 8
savedModelName = "model02.h5"

# load data
inputFrames = []
outputFrames = []

# start reading simSimple 1000 ff.
for sim in range(1000,2000): 
	dir = "{}/simSimple_{:04d}/".format(basePath,sim)
	if not os.path.exists( dir ):
		break

	inFrames, outFrames, stepsPerSim, simRes = ioext.loadTimeseries(dir + "density_*.uni", dir + "vel_*.uni", timeFrame)
	inputFrames.extend(inFrames)
	outputFrames.extend(outFrames)

inSize         = simRes[0] * simRes[1] * simRes[2] # warning - hard coded to scalar values 64^2
seriesPerSim = stepsPerSim - timeFrame



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
trainData = trainData.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

validData = tf.data.Dataset.from_tensor_slices((validationIn, validationOut))
validData = validData.batch(BATCH_SIZE)


# set up the network
model = keras.models.Sequential([
	layers.LSTM(64, input_shape=inputFrames.shape[-2:], activation='sigmoid'), # default tanh throws error "Skipping optimization due to error while loading"
	layers.Dense(inSize)
#	layers.Reshape((64,64,4))
])

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop())

# now we can start training...
print("Starting training...")
history = model.fit(trainData,
                    epochs=3,
                    validation_data=validData,
					steps_per_epoch=inputFrames.shape[0])

model.save("savedModelName.h5")
print("Saved model.")

