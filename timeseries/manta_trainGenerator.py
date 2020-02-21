#******************************************************************************
#
# Simple randomized sim data generation
#
#******************************************************************************
from manta import *
import os
import shutil
import math
import sys
import time
import argparse
import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import frequency
import tfextensions
sys.path.append("../utils")
import gridext
import ioext

# Main params
# ----------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="Train and generate data simultaneously.")
parser.add_argument('--steps', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resolution', type=int, default=64)
parser.add_argument('--modelName', default="vorticity3.h5")
parser.add_argument('--gui', dest='showGui', action='store_true')
parser.set_defaults(showGui=False)

args = parser.parse_args()

steps = args.steps
simName = "NeuralTurbolance"
npSeed = 1282749#args.seed #9782341
resolution = args.resolution

# Scene settings
# ---------------------------------------------------------------------#
setDebugLevel(0)
# in this mode no network is trained
writeData = False
dataSetSize = 1024

# NN params
# ----------------------------------------------------------------------#
modelName = "scale8InflowInp"
windowSize = 8
batchSize = 4

batchDistance = 1024

inOutScale = 8

class Format(Enum):
	SPATIAL = 1
	FREQUENCY = 2

class Inputs(Enum):
	VORTICITY = 1
	INFLOW = 2

	def asOut(self):
		return "OUT_" + self.name

inputFormat = Format.FREQUENCY
outputFormat = Format.FREQUENCY
# only expect low frequency components
useReducedOutput = True
useLagWindows = False
useInflowInput = True

# Solver params
# ----------------------------------------------------------------------#
res = resolution
dim = 2 
offset = 20
interval = 1
useMovingObstacle = False
useVaryingInflow = False
useVaryingNoise = True
varyNoiseSteps = 2048

resIn = res // inOutScale
simResRed = (res,res*2)
simRes = simResRed + (1,)

if inputFormat == Format.FREQUENCY:
	# last dimension for real, imag part
	lowFreqRes = (resIn,resIn+1,2)
elif inputFormat == Format.SPATIAL:
	lowFreqRes = (resIn,simRes[1]//inOutScale,1)

if outputFormat == Format.FREQUENCY:
	outputRes = (res*(res+1) - resIn*(resIn+1), 2, 1)
elif outputFormat == Format.SPATIAL:
	outputRes = simRes
	
gs = vec3(res,res, 1 if dim == 2 else res)
buoy = vec3(0,-1e-3,0)

sm = Solver(name='smaller', gridSize = simRes, dim=dim)
sm.timestep = 0.5

timings = Timings()

# Simulation Grids
# -------------------------------------------------------------------#
flags = sm.create(FlagGrid)
vel = sm.create(MACGrid)
vorticity = sm.create(RealGrid)
density = sm.create(RealGrid)
pressure = sm.create(RealGrid)
divergence = sm.create(RealGrid)
#obsVel  = s.create(MACGrid)
velReconstructed = sm.create(MACGrid)

# open boundaries
bWidth = 1
flags.initDomain(boundaryWidth=bWidth)

#obsPos = gs*vec3(0.5,0.4,0.5)
#obsVel.setBound(value=Vec3(0.), boundaryWidth=bWidth+1) # make sure walls are static
#obs = "dummy"; phiObs = "dummy2"
obstacle = sm.create(Sphere, center=gs*vec3(0.5,0.4,0.5), radius=res*0.10)
phiObs = obstacle.computeLevelset()
setObstacleFlags(flags=flags, phiObs=phiObs)

flags.fillGrid()
setOpenBound(flags,	bWidth,'xXyY',FlagOutflow | FlagEmpty) 
# inflow sources
# ----------------------------------------------------------------------#
if(npSeed != 0): np.random.seed(npSeed)


# inflow noise field
def makeNoiseField():
	noise = NoiseField( parent=sm, fixedSeed = np.random.randint(2**30), loadFromFile=True)
	noise.posScale = vec3(45)
	noise.clamp = True
	noise.clampNeg = 0
	noise.clampPos = 1
	noise.valOffset = 0.75
	noise.timeAnim = 0.2

	return noise

source     = Cylinder( parent=sm, center=gs*vec3(0.5,0.0,0.5), radius=res*0.081, z=gs*vec3(0, 0.1, 0))
sourceSize = (int(math.ceil(res*0.081))*2, int(math.ceil(res*0.1)))
sourcePos  = (int(res*0.5) - sourceSize[0] // 2, 1)
#sourceVel = Cylinder( parent=sm, center=gs*vec3(0.5,0.2,0.5), radius=res*0.15, z=gs*vec3(0.05, 0.0, 0))

if args.showGui:
	gui = Gui()
	gui.show()

# put enum in global namespace for shorter identifiers
globals().update(Inputs.__members__)

# simulation
def moveWindow(window, entry):
	window[0:-1] = window[1:]
	window[-1] = entry

def generateData(offset, batchSize):
	noise = makeNoiseField()

	historySize = batchSize * batchDistance
	slidingWindow = { Inputs.VORTICITY : np.zeros((windowSize,)+lowFreqRes)}
	# ringbuffer of previous states to create batches from
	inputHistory = { Inputs.VORTICITY : np.zeros((historySize,)+slidingWindow[Inputs.VORTICITY].shape) }
	outputHistory = { Inputs.VORTICITY : np.zeros((historySize,)+outputRes)}
	if(useInflowInput):
		slidingWindow[INFLOW] = np.zeros((windowSize,)+sourceSize)
		inputHistory[INFLOW]  = np.zeros((historySize,)+slidingWindow[INFLOW].shape)
	lowPass = np.array(lowFreqRes)[0:2]
	historyPtr = 0
	velInflow = vec3(np.random.uniform(-0.02,0.02), 0, 0)

	t = 0
	beginYield = offset + historySize * (1 if useLagWindows else windowSize)
	# main loop
	# --------------------------------------------------------------------#
	while 1:
		curt = t * sm.timestep
		mantaMsg("Current time t: " + str(curt) + " \n")
	
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True, boundaryWidth=bWidth)
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,	 order=2, openBounds=True, boundaryWidth=bWidth)

		if useInflowInput:
			preDensity = gridext.toNumpyArray(density,simResRed)
			np.save("preDensity", preDensity)
		densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
		if useVaryingInflow:
			if t % 867:
				velInflow = vec3(np.random.uniform(-0.01,0.01), 0, 0)
			source.applyToGrid( grid=vel , value=(velInflow*float(res)) )

	#	resetOutflow( flags=flags, real=density )

	#	vorticityConfinement(vel=vel, flags=flags, strength=0.05)
		setWallBcs(flags=flags, vel=vel)
		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)
	#	if (t < offset): 
	#		vorticityConfinement(vel=vel, flags=flags, strength=0.05)
		solvePressure(flags=flags, vel=vel, pressure=pressure ,  cgMaxIterFac=10.0, cgAccuracy=0.0001)

		getCurl(vel=vel, vort=vorticity , comp=2)
	#	getCodifferential(vel=velReconstructed, vort=vorticity)

	#	getDivergence(div=divergence, vel=velReconstructed)
	#	print("div reconstructed: {}".format(divergence.getL2()))
	#	getDivergence(div=divergence, vel=vel)
	#	print("div original:      {}".format(divergence.getL2()))
	#	divergence.setConst(0)

	#	velReconstructed.sub(vel)
	#	print("dif vectorfields:  {}".format(velReconstructed.getL2()))
	#	velReconstructed.setConst(vec3(0,0,0))
#		vorticity.printGrid()

		currentVal = gridext.toNumpyArray(vorticity,simResRed)
		currentVal = currentVal[:,::-1]
		
		freqs, lowFreqs = frequency.decomposeReal(currentVal, lowPass)
		if inputFormat == Format.SPATIAL:
			input = frequency.invTransformReal(lowFreqs)
		else:
			input = lowFreqs
		input = np.reshape(input, lowFreqRes)

		if useReducedOutput:
			currentVal = frequency.shrink(freqs, lowPass)
			if outputFormat == Format.SPATIAL:
				currentVal = frequency.invTransformReal(frequency.composeReal(currentVal, np.zeros(lowFreqs.shape), freqs.shape))
		else:
			if outputFormat == Format.SPATIAL:
				currentVal = frequency.invTransformReal(currentVal)
		currentVal = np.reshape(currentVal, outputRes)

		# update sliding windows
		moveWindow(slidingWindow[VORTICITY], input)
		if useInflowInput:
			postDensity = gridext.toNumpyArray(density,simResRed)
			dif = postDensity - preDensity
			inflow = dif[sourcePos[0]:sourcePos[0]+sourceSize[0], sourcePos[1]:sourcePos[1]+sourceSize[1]]
			moveWindow(slidingWindow[INFLOW], inflow)
		
		# record history
		if t > offset and (t % windowSize == 0 or useLagWindows):
			historyPtr = (historyPtr + 1) % historySize
			for key in inputHistory:
				inputHistory[key][historyPtr] = slidingWindow[key]
			
			outputHistory[VORTICITY][historyPtr] = currentVal

			# create data batches after history is full
			if t > beginYield:

				if writeData and t > 409600 and t < 410600:
					np.save("{}/lowres_{:04d}".format(dataPath,t-409600), input)
					np.save("{}/fullres_{:04d}".format(dataPath,t-409600), currentVal)

				# create input/output batches from history
				inputs = {}
				for key in inputHistory:
					currentBatch = []
					for i in range(batchSize):
						index = (historyPtr + i * batchDistance) % historySize
						currentBatch.append(np.copy(inputHistory[key][index]))
					inputs[key.name] = np.reshape(currentBatch, (batchSize,)+inputHistory[key][0].shape)
				outputs = {}
				for key in outputHistory:
					currentBatch = []
					for i in range(batchSize):
						index = (historyPtr + i * batchDistance) % historySize
						currentBatch.append(np.copy(outputHistory[key][index]))
					outputs[key.asOut()] = np.reshape(currentBatch, (batchSize,)+outputHistory[key][0].shape)
				yield (inputs, outputs)

		if useVaryingNoise and t % varyNoiseSteps == 0:
			noise.posOffset = vec3(np.random.random()*128,np.random.random()*128,np.random.random()*128)

		sm.step()
		t = t + 1

if writeData:
	dataPath = "data/long_{}_{}_{}_{}_{}".format(inOutScale, inputFormat.name[0], outputFormat.name[0], useReducedOutput, npSeed)
	if not os.path.exists(dataPath):
		os.makedirs(dataPath)
		gen = generateData(1024, 1)
		for i in range(dataSetSize):
			next(gen)
	else:
		print("Folder already exists. No data will be written.")
	exit()

# model setup
# ----------------------------------------------------------------------#
print("Setting up model.")

def buildModel(batchSize, windowSize):
	outputSize = outputRes[0]*outputRes[1]*outputRes[2]
	inSize = lowFreqRes[0] * lowFreqRes[1] * lowFreqRes[2]

	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	inflowInput    = keras.Input(shape=(windowSize,)+sourceSize,
					  batch_size=batchSize, name=INFLOW.name)
#	x = layers.TimeDistributed(layers.Conv2D(32,2,2))(inputs)
#	x = layers.TimeDistributed(layers.Conv2D(16,2,2))(x)
	flatInputVort = layers.Reshape((windowSize, inSize))(vorticityInput)
	flatInputInFlow = layers.Reshape((windowSize, sourceSize[0]*sourceSize[1]))(inflowInput)
	flatInput = layers.Concatenate(axis=2)([flatInputVort, flatInputInFlow])
#	first = layers.Dense(512)(flatInput)
	first = layers.LSTM(144, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
#	x1 = layers.Add()([first,flatInput])
	x2 = layers.LSTM(144, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
#	x2 = layers.LSTM(64, stateful=True, return_sequences=True)(x1)
#	x1 = layers.Add()([x1,x2])
#	x2 = layers.LSTM(144, stateful=True, return_sequences=True)(x1)
#	x1 = layers.Add()([x1,x2])
	x1 = layers.LSTM(256, stateful=True, return_sequences=False)(x1)
	x2 = layers.Dense(256, activation='tanh')(x1)
#	x2 = layers.BatchNormalization()(x2)
	x1 = layers.Add()([x1,x2])
	x = layers.Dense(512, activation='tanh')(x1)
#	x = layers.Dense(1024, activation='tanh')(x)
#	x = layers.BatchNormalization()(x)
#	x2 = layers.BatchNormalization()(x2)
#	x1 = layers.Add()([x1,x2])
#	x2 = layers.LSTM(128, stateful=True, return_sequences=True)(x1)
#	x1 = layers.Add()([x1,x2])
#	x = layers.LSTM(512, stateful=True, return_sequences=False)(x1)
#	first = layers.Lambda(lambda x : x[:,-1])(first)
#	x = layers.Add()([first,x])
#	x = layers.LSTM(512, stateful=True, return_sequences=False)(x)
#	x = layers.Reshape((1,1,512))(first)
#	x = layers.Conv2DTranspose(128, (2,4), strides=(1,1))(x)
#	x = layers.Conv2DTranspose(64, 2, strides=(2,2), padding='same')(x)
#	x = layers.Conv2DTranspose(32, 4, strides=(2,2), padding='same')(x)
#	x = layers.Conv2DTranspose(32, 5, strides=(2,2), padding='same')(x)
#	x = layers.Conv2DTranspose(16, 5, strides=(2,2), padding='same')(x)
#	output = layers.Conv2DTranspose(1, 5, strides=(2,2), padding='same')(x)
#	y = layers.LSTM(inSize, stateful=True)(flatInput)
#	x = layers.concatenate([x,y])
#	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	# extract current time input
#	forward = layers.Reshape((lowFreqRes))(inputs)
#	forward = layers.Lambda(lambda x : x[:,-1])(inputs)
#	forward = tfextensions.UpsampleFreq(lowFreqRes, outputRes)(forward)
#	forward = layers.UpSampling2D(interpolation='bilinear')(forward)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
#	output = layers.Add()([forward, output])
	model = keras.Model(inputs=[vorticityInput, inflowInput], outputs=output)

	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(),
			metrics=[tfextensions.frequencyLoss])
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model

model = buildModel(batchSize,windowSize)
#model = tf.keras.models.load_model("learning.h5", tfextensions.functionMap)

# model training
# ----------------------------------------------------------------------#

# validation data set
if outputFormat == Format.FREQUENCY:
	if inputFormat == Format.FREQUENCY:
		dataName = "val_8_F_F_True_9782341"
	else:
		dataName = "val_8_S_F_True_9782341"
else:
	if inputFormat == Format.FREQUENCY:
		dataName = "val_8_F_S_True_9782341"
	else:
		dataName = "val_8_S_S_True_9782341"
path = "data/" + dataName + "/"

inputs = ioext.loadNPData(path + "lowres_*.npy")
outputs = ioext.loadNPData(path + "fullres_*.npy")
inputFrames, outputs, _ = ioext.createTimeSeries(inputs, outputs, 1, useLagWindows)
inputFrames = np.reshape(inputFrames, (len(inputFrames), 1)+lowFreqRes)
outputs = np.reshape(outputs, (len(outputs), )+outputRes)
#if batchSize > 1:
#	inputFrames, outputs = ioext.createBatches(inputFrames, outputs, batchSize)

def generateValid(inputBatches, outputBatches):
	ind = 0
	while 1:
		yield (inputBatches[ind], outputBatches[ind])
		ind = (ind + 1) % len(inputBatches)

class ValidationCallback(tf.keras.callbacks.Callback):
	def __init__(self, inputs, outputs, testModel, runFrequency):
		super().__init__()

		self.testModel = testModel
		self.inputs = inputs
		self.outputs = outputs
		self.frequency = runFrequency
		self.history = []

	def on_epoch_end(self, epoch,logs=None):
		if(epoch % self.frequency != 0): 
			return
		self.testModel.set_weights(self.model.get_weights())
		self.testModel.reset_states()
	#	self.model.reset_states()
		outputs = self.testModel.predict(self.inputs)
		sum = 0
		for i in range(len(outputs)):
			dif = np.subtract(frequency.flattenComplex(outputs[i]), frequency.flattenComplex(self.outputs[i]))
			sum += np.mean(np.abs(dif))

		print("Validation error: {}".format(sum / len(outputs)))
		self.history.append(sum / len(outputs))

	def on_train_end(self, logs=None):
		arr = np.reshape(self.history, (len(self.history),))
		np.save("validLog5",arr)


print("Starting training.")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="currentmodel/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)
validation_callback = ValidationCallback(inputFrames, outputs, buildModel(1,1), 1)

history = model.fit_generator(generateData(1024, batchSize),
							  steps_per_epoch=512, 
							  epochs=128,
							#  validation_freq=1,
							#  validation_data=generateValid(inputFrames, outputs),
							#  validation_steps=len(outputs),
							  callbacks=[cp_callback, validation_callback],
							  use_multiprocessing=False)

model.save("learningTest.h5")
modelS = buildModel(1,1)
modelS.set_weights(model.get_weights())
fullName = "{}_W{}_B{}.h5".format(modelName, windowSize, batchSize)
modelS.save(fullName)
print("Saved model as {}.".format(fullName))