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
npSeed = 1282749#args.seed #9782341 #1282749 #5281034 # 3570583
resolution = args.resolution

# Scene settings
# ---------------------------------------------------------------------#
setDebugLevel(0)
# in this mode no network is trained
writeData = True
dataSetSize = 1024
dataName = "all"

# NN params
# ----------------------------------------------------------------------#
modelName = "AddInputs"
windowSize = 1
batchSize = 1

batchDistance = 1024

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
varyScalingSteps = 1277
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
	slidingWindow = { VORTICITY : np.zeros((windowSize,)+lowFreqRes)}
	# ringbuffer of previous states to create batches from
	inputHistory = { VORTICITY : np.zeros((historySize,)+slidingWindow[Inputs.VORTICITY].shape) }
	outputHistory = { VORTICITY : np.zeros((historySize,)+outputRes)}
	# optional inputs
	if useInflowInput:
		slidingWindow[INFLOW] = np.zeros((windowSize,)+(sourceSize if not useFullResInflow else outputResFull))
		inputHistory[INFLOW]  = np.zeros((historySize,)+slidingWindow[INFLOW].shape)
	if useInflowVelInput:
		velRes = (sourceSize if not useFullResInflow else outputResFull)
		slidingWindow[INFLOWVEL] = np.zeros((windowSize,)+velRes+(2,)) # 2 channels for the vector field
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
	velInflow = vec3(np.random.uniform(-0.02,0.02), 0, 0)

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
			if t % 863 == 0:
				velInflow = vec3(np.random.uniform(-0.01,0.01), 0, 0)
			source.applyToGrid( grid=vel , value=(velInflow*float(res)) )

		setWallBcs(flags=flags, vel=vel)
		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)

		if useScalingObstacle and t % varyScalingSteps == 0:
			obsCurRadius = obsRad*np.random.uniform(0.90,1.10)
			obstacle = sm.create(Sphere, center=obsPos, radius=obsCurRadius)
			phiObs = obstacle.computeLevelset()
			setObstacleFlags(flags=flags, phiObs=phiObs)
			flags.fillGrid()
			obs.applyToGrid(grid=density, value=0.)

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
			velocity = gridext.toNumpyArray(vel,simResRed+(2,))
			inflow = velocity[sourcePos[0]:sourcePos[0]+sourceSize[0], sourcePos[1]:sourcePos[1]+sourceSize[1]] if not useFullResInflow else np.reshape(dif[:,::-1], outputResFull)
			moveWindow(slidingWindow[INFLOWVEL], inflow)
		if useObstacleInput:
			moveWindow(slidingWindow[OBSTACLE], obsCurSize)
		
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
if writeData:
	dataPath = "data/{}_{}_{}_{}_{}_{}".format(dataName, inOutScale, inputFormat.name[0], outputFormat.name[0], useReducedOutput, npSeed)
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
	flatInput = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInput = layers.TimeDistributed(layers.Dense(128))(flatInput)
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
	model = keras.Model(inputs=[vorticityInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model

def buildModel2(batchSize, windowSize):
	outputSize = outputRes[0]*outputRes[1]*outputRes[2]
	inSize = lowFreqRes[0] * lowFreqRes[1] * lowFreqRes[2]

	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	flatInput = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInput = layers.TimeDistributed(layers.Dense(128))(flatInput)
	first = layers.LSTM(40, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(40, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
	x1 = layers.LSTM(40, stateful=True, return_sequences=False)(x1)
	x = layers.Reshape((1,1,40))(x1)
	x = layers.Dense(256, activation='tanh')(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
	model = keras.Model(inputs=[vorticityInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model

def buildModel3(batchSize, windowSize):
	outputSize = outputRes[0]*outputRes[1]*outputRes[2]
	inSize = lowFreqRes[0] * lowFreqRes[1] * lowFreqRes[2]

	vorticityInput = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize, name=VORTICITY.name)
	flatInput = layers.Reshape((windowSize, 128))(vorticityInput)
	flatInput = layers.TimeDistributed(layers.Dense(128))(flatInput)
	first = layers.LSTM(40, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(40, stateful=True, return_sequences=True)(first)
	x1 = layers.Add()([first,x2])
	x1 = layers.LSTM(40, stateful=True, return_sequences=False)(x1)
#	x = layers.Reshape((40,))(x1)
	flatInput = layers.Lambda(lambda x : x[:,-1])(flatInput)
	x = layers.Concatenate(axis=1)([flatInput, x1])
	x = layers.Dense(256, activation='tanh')(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.Dense(outputSize)(x)
	output = layers.Reshape(outputRes, name=VORTICITY.asOut())(x)
	model = keras.Model(inputs=[vorticityInput], outputs=output)
	if outputFormat == Format.SPATIAL:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
	else:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop())

	return model


models = [buildModel(batchSize,windowSize)]#, buildModel2(batchSize,windowSize), buildModel3(batchSize,windowSize)]
validModels = [buildModel(1,1)]#, buildModel2(1,1),buildModel3(1,1)]
models[0].summary()
#model = tf.keras.models.load_model("learning.h5", tfextensions.functionMap)

# model training
# ----------------------------------------------------------------------#
fullName = "{}_W{}_B{}".format(modelName, windowSize, batchSize)

# validation data set
if outputFormat == Format.FREQUENCY:
	if inputFormat == Format.FREQUENCY:
		dataName = "fullout_8_F_F_True_5281034"
	else:
		dataName = "fullout_8_S_F_True_5281034"
else:
	if inputFormat == Format.FREQUENCY:
		dataName = "fullout_8_F_S_True_5281034"
	else:
		dataName = "fullout_8_S_S_True_5281034"
path = "data/" + dataName + "/"

inputs, outputs, _ = loadDataSet(path, 1, useLagWindows);
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

generator = generateData(1024, batchSize)
for i in range(len(models)):
	validation_callback = ValidationCallback(inputs, outputs, validModels[i], 1)
	history = models[i].fit(x = generator,
							  steps_per_epoch=512, 
							  epochs=50,
							  callbacks=[validation_callback], # validation_callback
							  use_multiprocessing=False)

#model.save("learningTest.h5")
	modelS = validModels[i]
	fullName = "{}_".format(fullName, i)
	modelS.set_weights(models[i].get_weights())
	saveName = fullName + ".h5"
	modelS.save(saveName)
	print("Saved model as {}.".format(saveName))