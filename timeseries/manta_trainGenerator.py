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
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import frequency
import tfextensions
sys.path.append("../utils")
import gridext
import ioext
from common import *

# Main params
# ----------------------------------------------------------------------#

simName = "NeuralTurbolance"
npSeed = 3570583#args.seed #9782341 #1282749 #5281034 # 3570583
resolution = 64

showGui = False
writeData = False # in this mode no network is trained
dataSetSize = 1024
dataName = "allInActive"
warmupSteps = 1024

# Scene settings
# ---------------------------------------------------------------------#
setDebugLevel(0)

# NN params
# ----------------------------------------------------------------------#
modelName = "AddInputsAll3"
windowSize = 1
batchSize = 1

batchDistance = 512

inOutScale = 8

inputFormat = Format.SPATIAL
outputFormat = Format.FREQUENCY
# only expect low frequency components
useReducedOutput = True
useLagWindows = False
useInflowInput = True
useFullResInflow = False
usePreviousStep = False
useObstacleInput = True
useInflowVelInput = True

# Solver params
# ----------------------------------------------------------------------#
res = resolution
dim = 2 
offset = 20
interval = 1
useMovingObstacle = False
useScalingObstacle = False
varyScalingSteps = 277#947
useVaryingInflow = False
varyInflowSteps = 357
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
	outputResFull = (res*(res+1), 2, 1)
elif outputFormat == Format.SPATIAL:
	outputRes = simRes
	outputResFull = simRes
	
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

obsPos = gs*vec3(0.5,0.4,0.5)
obsRad = res*0.1
obsCurRadius = obsRad
obstacle = sm.create(Sphere, center=obsPos, radius=obsRad)
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

if showGui:
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
	slidingWindow = { VORTICITY : np.zeros((windowSize,)+lowFreqRes)}
	# ringbuffer of previous states to create batches from
	inputHistory = { VORTICITY : np.zeros((historySize,)+slidingWindow[Inputs.VORTICITY].shape) }
	outputHistory = { VORTICITY : np.zeros((historySize,)+outputRes)}
	# optional inputs
	if useInflowInput:
		slidingWindow[INFLOW] = np.zeros((windowSize,)+(sourceSize if not useFullResInflow else outputResFull))
		inputHistory[INFLOW]  = np.zeros((historySize,)+slidingWindow[INFLOW].shape)
	if useInflowVelInput:
		slidingWindow[INFLOWVEL] = np.zeros((windowSize,2))
		inputHistory[INFLOWVEL]  = np.zeros((historySize,)+slidingWindow[INFLOWVEL].shape)
	if usePreviousStep:
		slidingWindow[PREVVORT] = np.zeros((windowSize,)+outputResFull)
		inputHistory[PREVVORT]  = np.zeros((historySize,)+slidingWindow[PREVVORT].shape)
		previousStep = np.zeros(outputResFull)
	if useObstacleInput:
		slidingWindow[OBSTACLE] = np.zeros((windowSize,1))
		inputHistory[OBSTACLE]  = np.zeros((historySize,)+slidingWindow[OBSTACLE].shape)
	lowPass = np.array((resIn,resIn+1))
	historyPtr = 0
	velInflowPy = (np.random.uniform(-0.004,0.004),0)
	velInflow = vec3(velInflowPy[0], 0, 0)
	obsCurRadius = obsRad

	t = 0
	start = time.time()
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
		densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
		if useVaryingInflow:
			if t % varyInflowSteps == 0:
				velInflowPy = (np.random.uniform(-0.002,0.002),0)
				velInflow = vec3(velInflowPy[0], 0, 0)
				velInflowPy = (velInflowPy[0] / 0.002, 0.0)
		#		print("{} inflow".format(t))
			source.applyToGrid( grid=vel , value=(velInflow*float(res)) )

		setWallBcs(flags=flags, vel=vel)
		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)

		if useScalingObstacle and t % varyScalingSteps == 0:
			obsCurRadius = obsRad*np.random.uniform(0.80,1.20)
			obstacle = sm.create(Sphere, center=obsPos, radius=obsCurRadius)
			phiObs = obstacle.computeLevelset()
			setObstacleFlags(flags=flags, phiObs=phiObs)
			flags.fillGrid()
			obstacle.applyToGrid(grid=density, value=0.)
		#	print("{} obstacle".format(t))

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

		if usePreviousStep:
			moveWindow(slidingWindow[PREVVORT], np.reshape(previousStep, outputResFull))
			if outputFormat == Format.SPATIAL:
				previousStep = currentVal
			else:
				previousStep = freqs
				

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
			if outputFormat == Format.FREQUENCY:
				currentVal = freqs
		currentVal = np.reshape(currentVal, outputRes)

		# update sliding windows
		moveWindow(slidingWindow[VORTICITY], input)
		if useInflowInput:
			postDensity = gridext.toNumpyArray(density,simResRed)
			dif = postDensity - preDensity
			inflow = dif[sourcePos[0]:sourcePos[0]+sourceSize[0], sourcePos[1]:sourcePos[1]+sourceSize[1]] if not useFullResInflow else np.reshape(dif[:,::-1], outputResFull)
			moveWindow(slidingWindow[INFLOW], inflow)
		if useInflowVelInput:
			moveWindow(slidingWindow[INFLOWVEL], velInflowPy)
		if useObstacleInput:
			moveWindow(slidingWindow[OBSTACLE], obsCurRadius / obsRad - 1.0)
		
		# record history
		if t > offset and (t % windowSize == 0 or useLagWindows):
			historyPtr = (historyPtr + 1) % historySize
			for key in inputHistory:
				inputHistory[key][historyPtr] = slidingWindow[key]
			
			outputHistory[VORTICITY][historyPtr] = currentVal

			# create data batches after history is full
			if t > beginYield:

				if writeData:
					for key,value in slidingWindow.items():
						np.save("{}/{}_{:04d}".format(dataPath,key.name,t), value[-1])
					np.save("{}/{}_{:04d}".format(dataPath,VORTICITY.asOut(), t), currentVal)

				# create input/output batches from history
				inputs = {}
				for key in inputHistory:
					currentBatch = np.zeros((batchSize,)+inputHistory[key][0].shape)
					for i in range(batchSize):
						index = (historyPtr + i * batchDistance) % historySize
						currentBatch[i] = inputHistory[key][index]
					inputs[key.name] = currentBatch
				outputs = {}
				for key in outputHistory:
					currentBatch = np.zeros((batchSize,)+outputHistory[key][0].shape)
					for i in range(batchSize):
						index = (historyPtr + i * batchDistance) % historySize
						currentBatch[i] = outputHistory[key][index]
					outputs[key.asOut()] = currentBatch
				yield (inputs, outputs, [None])
			elif t % 1000 == 0: # progress bar for warmup
				print(t / beginYield * 100)

		if useVaryingNoise and t % varyNoiseSteps == 0:
			noise.posOffset = vec3(np.random.random()*128,np.random.random()*128,np.random.random()*128)

		sm.step()
		t = t + 1
#		if t % 1000 == 0:
#			end = time.time()
#			print(end - start)
#			start = time.time()

#gen = generateData(1024,1)
#for i in range(dataSetSize):
#	next(gen)

if writeData:
	dataPath = "data/{}_{}_{}_{}_{}_{}".format(dataName, inOutScale, inputFormat.name[0], outputFormat.name[0], useReducedOutput, npSeed)
	if not os.path.exists(dataPath):
		os.makedirs(dataPath)
		gen = generateData(warmupSteps, 1)
		for i in range(dataSetSize):
			next(gen)
	else:
		print("Folder already exists. No data will be written.")
	exit()

# model setup
# ----------------------------------------------------------------------#
print("Setting up model.")

outputSize = outputRes[0]*outputRes[1]*outputRes[2]

def buildModel(batchSize, windowSize):
	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	inflowInput    = keras.Input(shape=(windowSize,)+sourceSize,
					  batch_size=batchSize, name=INFLOW.name)
	flatInputVort = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInputVort = layers.TimeDistributed(layers.Dense(128))(flatInputVort)
	flatInputInflow = layers.Reshape((windowSize, 12*7))(inflowInput)
	flatInputInflow = layers.TimeDistributed(layers.Dense(12*7))(flatInputInflow)
	flatInput = layers.Concatenate(axis=2)([flatInputVort, flatInputInflow])
	first = layers.LSTM(80, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(80, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
	x1 = layers.LSTM(80, stateful=True, return_sequences=False)(x1)
	x = layers.Reshape((1,1,80))(x1)
	x = layers.Dense(256, activation='tanh')(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
	model = keras.Model(inputs=[vorticityInput, inflowInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model

def buildModel2(batchSize, windowSize):
	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	inflowInput    = keras.Input(shape=(windowSize,)+sourceSize,
					  batch_size=batchSize, name=INFLOW.name)
	obstacleInput = keras.Input(shape=(windowSize,1),
					  batch_size=batchSize, name=OBSTACLE.name)
	inflowVelInput = keras.Input(shape=(windowSize,2),
					  batch_size=batchSize, name=INFLOWVEL.name)
	flatInputVort = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInputVort = layers.TimeDistributed(layers.Dense(128))(flatInputVort)
	flatInputInflow = layers.Reshape((windowSize, 12*7))(inflowInput)
	flatInputInflow = layers.TimeDistributed(layers.Dense(12*7))(flatInputInflow)
	flatInputObstacle = layers.LSTM(4, stateful=True, return_sequences=True)(obstacleInput)
	flatInputInflowVel = layers.LSTM(4, stateful=True, return_sequences=True)(inflowVelInput)
	flatInput = layers.Concatenate(axis=2)([flatInputVort, flatInputInflow, flatInputObstacle, flatInputInflowVel])
	first = layers.LSTM(80, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(80, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
	x1 = layers.LSTM(80, stateful=True, return_sequences=False)(x1)
	x = layers.Reshape((1,1,80))(x1)
	x = layers.Dense(256, activation='tanh')(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
	model = keras.Model(inputs=[vorticityInput, inflowInput, obstacleInput, inflowVelInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model
def buildModel3(batchSize, windowSize):
	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	inflowInput    = keras.Input(shape=(windowSize,)+sourceSize,
					  batch_size=batchSize, name=INFLOW.name)
	obstacleInput = keras.Input(shape=(windowSize,1),
					  batch_size=batchSize, name=OBSTACLE.name)
	inflowVelInput = keras.Input(shape=(windowSize,2),
					  batch_size=batchSize, name=INFLOWVEL.name)
	flatInputVort = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInputVort = layers.TimeDistributed(layers.Dense(128))(flatInputVort)
	flatInputInflow = layers.Reshape((windowSize, 12*7))(inflowInput)
	flatInputInflow = layers.TimeDistributed(layers.Dense(12*7))(flatInputInflow)
	flatInputObstacle = layers.Reshape((windowSize, 1))(obstacleInput)
	flatInputObstacle = layers.LSTM(4, stateful=True, return_sequences=True)(flatInputObstacle)
	flatInputInflowVel = layers.Reshape((windowSize, 2))(inflowVelInput)
	flatInputInflowVel = layers.LSTM(4, stateful=True, return_sequences=True)(flatInputInflowVel)
	flatInput = layers.Concatenate(axis=2)([flatInputVort, flatInputInflow, flatInputObstacle, flatInputInflowVel])
	first = layers.LSTM(100, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(100, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
	x2 = layers.LSTM(100, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([x1,x2])
	x2 = layers.LSTM(100, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([x1,x2])
	x1 = layers.LSTM(128, stateful=True, return_sequences=False)(x1)
	x = layers.Reshape((1,1,128))(x1)
	x = layers.Dense(256, activation='tanh')(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
	model = keras.Model(inputs=[vorticityInput, inflowInput, obstacleInput, inflowVelInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model

def buildModel4(batchSize, windowSize):
	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	inflowInput    = keras.Input(shape=(windowSize,)+sourceSize,
					  batch_size=batchSize, name=INFLOW.name)
	obstacleInput = keras.Input(shape=(windowSize,1),
					  batch_size=batchSize, name=OBSTACLE.name)
	flatInputVort = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInputVort = layers.TimeDistributed(layers.Dense(128))(flatInputVort)
	flatInputInflow = layers.Reshape((windowSize, 12*7))(inflowInput)
	flatInputInflow = layers.TimeDistributed(layers.Dense(12*7))(flatInputInflow)
	flatInputObstacle = layers.Reshape((windowSize, 1))(obstacleInput)
	flatInputObstacle = layers.LSTM(8, stateful=True, return_sequences=True)(flatInputObstacle)
	flatInput = layers.Concatenate(axis=2)([flatInputVort, flatInputInflow, flatInputObstacle])
	first = layers.LSTM(100, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(100, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
	x2 = layers.LSTM(100, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([x1,x2])
	x2 = layers.LSTM(100, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([x1,x2])
	x1 = layers.LSTM(128, stateful=True, return_sequences=False)(x1)
	x = layers.Reshape((1,1,128))(x1)
	x = layers.Dense(256, activation='tanh')(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
	model = keras.Model(inputs=[vorticityInput, inflowInput, obstacleInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model


models = [buildModel(batchSize,windowSize),
		  buildModel2(batchSize,windowSize)]
	#	  buildModel3(batchSize,windowSize)]
	#	  buildModel4(batchSize,windowSize)]
validModels = [buildModel(1,1),
			   buildModel2(1,1)]
	#		   buildModel3(1,1)]
	#		   buildModel4(1,1)]

# model training
# ----------------------------------------------------------------------#
fullName = "{}_W{}_B{}".format(modelName, windowSize, batchSize)

# validation data set
if outputFormat == Format.FREQUENCY:
	if inputFormat == Format.FREQUENCY:
		dataName = "all2_8_F_F_True_5281034"
	else:
		dataName = "allActive_8_S_F_True_1282749"
else:
	if inputFormat == Format.FREQUENCY:
		dataName = "all2_8_F_S_True_5281034"
	else:
		dataName = "all2_8_S_S_True_5281034"
path = "data/" + dataName + "/"

inputs, outputs, _ = loadDataSet(path, 1, useLagWindows);

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
		self.outputs = outputs[VORTICITY.asOut()]
		if(outputFormat == Format.SPATIAL):
			outputs = []
			for i in range(len(self.outputs)):
				outputs.append(frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(self.outputs[i]), axes=0)))
			self.outputs = np.reshape(outputs, (len(outputs),)+outputs[0].shape)
		self.frequency = runFrequency
		self.history = []

	def on_epoch_end(self, epoch,logs=None):
		if(epoch % self.frequency != 0): 
			return
		self.testModel.set_weights(self.model.get_weights())
		self.testModel.reset_states()
	#	self.model.reset_states()
		outputs = self.testModel.predict(self.inputs)
		if(outputFormat == Format.SPATIAL):
			outp = []
			for i in range(len(outputs)):
				outp.append(frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(outputs[i]), axes=0)))
		else:
			outp = outputs
		sum = 0
		for i in range(len(outputs)):
			dif = np.subtract(frequency.flattenComplex(outp[i]), frequency.flattenComplex(self.outputs[i]))
			sum += np.mean(np.abs(dif))

		print("Validation error: {}".format(sum / len(outputs)))
		self.history.append(sum / len(outputs))

	def on_train_end(self, logs=None):
		arr = np.reshape(self.history, (len(self.history),))
		np.save(fullName,arr)


print("Starting training.")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="currentmodel/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

generator = generateData(warmupSteps, batchSize)
for i in range(len(models)):
	validation_callback = ValidationCallback(inputs, outputs, validModels[i], 1)
	history = models[i].fit(x = generator,
							  steps_per_epoch=12, 
							  epochs=5,
							  callbacks=[validation_callback], # validation_callback
							  use_multiprocessing=False)

#model.save("learningTest.h5")
	modelS = validModels[i]
	fullName = "{}_".format(fullName, i)
	modelS.set_weights(models[i].get_weights())
	saveName = fullName + ".h5"
	modelS.save(saveName)
	print("Saved model as {}.".format(saveName))