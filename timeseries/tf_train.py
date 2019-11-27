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
basePath = 'data/'
simName = "simSimple_"

trainingEpochs = 10
batchSize      = 10
timeFrame      = 4
convStride     = 2
lstmSize       = 4096
savedModelName = "model"

# load data
inputFrames = []
outputFrames = []
stepsPerSim = 0
simRes = (0,0,0)

# start reading simSimple 1000 ff.
for sim in range(1031,2000): 
	dir = "{}{}{:04d}/".format(basePath,simName,sim)
	if not os.path.exists( dir ):
		break

	inFrames, outFrames, stepsPerSim, simRes = ioext.loadTimeseries(dir + "density_*.uni", dir + "vel_*.uni", timeFrame)
	inputFrames.extend(inFrames)
	outputFrames.extend(outFrames)

inSize         = simRes[0] * simRes[1] * simRes[2]
seriesPerSim = stepsPerSim - timeFrame
validSize = min(seriesPerSim, 256)
print("Using training set of size {}.".format(len(inputFrames)))
print("Using validation set of size {}.".format(validSize))



validationIn = inputFrames[-validSize:];
validationOut = outputFrames[-validSize:];
validationIn = np.reshape(validationIn, (len(validationIn),timeFrame) + simRes)
validationOut = np.reshape(validationOut, (len(validationOut),) + simRes)

inputFrames = inputFrames[:-validSize];
outputFrames = outputFrames[:-validSize];
inputFrames = np.reshape(inputFrames, (len(inputFrames),timeFrame)+simRes)
outputFrames = np.reshape(outputFrames, (len(inputFrames),) + simRes)

BATCH_SIZE = 16
BUFFER_SIZE = 1000

trainData = tf.data.Dataset.from_tensor_slices((inputFrames, outputFrames))
trainData = trainData.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

validData = tf.data.Dataset.from_tensor_slices((validationIn, validationOut))
validData = validData.batch(BATCH_SIZE)

# set up the network
numFilters = 1#simRes[2]
lstmInSize = simRes[0]*simRes[1]*numFilters# // convStride**2
totalConv = convStride

model = keras.models.Sequential([
#	layers.TimeDistributed(layers.Conv2D(numFilters, 2, strides=convStride, padding='same'), input_shape=(timeFrame,)+simRes),
#	layers.TimeDistributed(layers.Conv2D(4, 2, strides=convStride, padding='same')),
	layers.Reshape((timeFrame,lstmInSize)),
	layers.LSTM(lstmSize, activation='sigmoid', stateful=False), # default tanh throws error "Skipping optimization due to error while loading"
#	layers.Dense(lstmInSize // 3),
#	layers.Reshape((simRes[0] // totalConv, simRes[1] // totalConv,1)),
#	layers.Conv2DTranspose(1,2,strides=convStride),
#	layers.Flatten(),
#	layers.Dense(simRes[0]*simRes[1]*simRes[2]),
#	layers.Conv2DTranspose(4,3,strides=1)
	layers.Reshape(simRes)
])

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop())

# now we can start training...
print("Starting training...")
history = model.fit(trainData,
                    epochs=trainingEpochs,
                    validation_data=validData,
					steps_per_epoch=256)

fileName = "{}.h5".format(savedModelName)
i = 0
while os.path.isfile(fileName):
	fileName = "{}{}.h5".format(savedModelName,i)
	i += 1
model.save(fileName)
print("Saved model as {}.".format(fileName))

