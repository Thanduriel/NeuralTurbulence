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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import frequency
import tfextensions
sys.path.append("../utils")
import gridext

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
simName = "trololo"
npSeed = args.seed
resolution = args.resolution

# Scene settings
# ---------------------------------------------------------------------#
setDebugLevel(0)

# NN params
# ----------------------------------------------------------------------#
windowSize = 1
lstmSize = resolution * resolution
batchSize = 4
learnFrequency = True

batchDistance = 1024
historySize = batchSize * batchDistance

# Solver params
# ----------------------------------------------------------------------#
res = resolution
dim = 2 
offset = 20
interval = 1

simResRed = (res,res*2)
simRes = simResRed + (1,)
if learnFrequency:
	outputRes = (res,res+1,2)
	# last dimension for real, imag part
	# y // 4: use fft symmetry for real signal
	lowFreqRes = (res//2,res//2+1,2)
else:
	outputRes = (res,res*2,1)
	lowFreqRes = (res//2,simRes[1]//2,1)
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
velReconstructed = sm.create(MACGrid)

# open boundaries
bWidth = 1
flags.initDomain(boundaryWidth=bWidth)
obstacle = sm.create(Sphere, center=gs*vec3(0.5,0.4,0.5), radius=res*0.10)
phiObs = obstacle.computeLevelset()
setObstacleFlags(flags=flags, phiObs=phiObs)
flags.fillGrid()
setOpenBound(flags,	bWidth,'xXyY',FlagOutflow | FlagEmpty) 

# inflow sources
# ----------------------------------------------------------------------#
if(npSeed != 0): np.random.seed(npSeed)

# note - world space velocity, convert to grid space later
velInflow = vec3(np.random.uniform(-0.02,0.02), 0, 0)

# inflow noise field
noise = NoiseField( parent=sm, fixedSeed = np.random.randint(2**30), loadFromFile=True)
noise.posScale = vec3(45)
noise.clamp = True
noise.clampNeg = 0
noise.clampPos = 1
noise.valOffset = 0.75
noise.timeAnim = 0.2

source    = Cylinder( parent=sm, center=gs*vec3(0.5,0.0,0.5), radius=res*0.081, z=gs*vec3(0, 0.1, 0))
#sourceVel = Cylinder( parent=sm, center=gs*vec3(0.5,0.2,0.5), radius=res*0.15, z=gs*vec3(0.05, 0.0, 0))

if args.showGui:
	gui = Gui()
	gui.show()

def generateData(offset, batchSize):
	slidingWindow = np.zeros((windowSize,)+lowFreqRes)
	# ringbuffer of previous states to create batches from
	inputHistory = np.zeros((historySize,)+slidingWindow.shape)
	outputHistory = np.zeros((historySize,)+outputRes)
	historyPtr = 0

	t = 0
	# main loop
	# --------------------------------------------------------------------#
	while 1:
		curt = t * sm.timestep
		mantaMsg("Current time t: " + str(curt) + " \n")
	
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True, boundaryWidth=bWidth)
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,	 order=2, openBounds=True, boundaryWidth=bWidth)

	#	if (sm.timeTotal>=0 and sm.timeTotal<offset):
		densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
		#	sourceVel.applyToGrid( grid=vel , value=(velInflow*float(res)) )

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
		freqs, lowFreqs = frequency.decomposeReal(currentVal, np.array(lowFreqRes)[0:2])
		#input = frequency.invTransformReal(lowFreqs)
		currentVal = freqs
		input = np.reshape(lowFreqs, lowFreqRes)
		currentVal = np.reshape(currentVal, outputRes)

		# move window
	#	slidingWindow[0:windowSize-1] = slidingWindow[1:]
		slidingWindow[-1] = input
		
		# record history
		if t > offset:
			historyPtr = (historyPtr + 1) % historySize
			inputHistory[historyPtr] = slidingWindow
			outputHistory[historyPtr] = currentVal

		# create data batches after history is full
		if t > offset+historySize:
			currentInputBatch = []
			currentOutputBatch = []
			for i in range(batchSize):
				index = (historyPtr + i * batchDistance) % historySize
				currentInputBatch.append(np.copy(inputHistory[index]))
				currentOutputBatch.append(np.copy(outputHistory[index]))

			inputs = np.reshape(currentInputBatch, (batchSize,)+slidingWindow.shape)
			outputs = np.reshape(currentOutputBatch, (batchSize,)+currentVal.shape)
			yield (inputs, outputs)

		#	np.save("data/vorticitySym3/lowres_{:04d}".format(t), input)
		#	np.save("data/vorticitySym3/fullres_{:04d}".format(t), currentVal)

		sm.step()
		t = t + 1

#gen = generateData(1024, batchSize)
#for i in range(512):
#	next(gen)
#exit()

# model setup
# ----------------------------------------------------------------------#
print("Setting up model.")

def buildModel(batchSize):
	lstmInSize = outputRes[0]*outputRes[1]*outputRes[2]
	inSize = lowFreqRes[0] * lowFreqRes[1] * lowFreqRes[2]

	inputs = keras.Input(shape=(windowSize,)+lowFreqRes,
					  batch_size=batchSize)
#	x = layers.TimeDistributed(layers.Conv2D(32,2,2))(inputs)
#	x = layers.TimeDistributed(layers.Conv2D(16,2,2))(x)
	flatInput = layers.Reshape((windowSize,inSize))(inputs)
	x1 = layers.LSTM(256, activation='tanh', 
				stateful=True,
				return_sequences=True)(flatInput)
	x2 = layers.LSTM(256, stateful=True)(x1)
	x1 = layers.Add()([x1,x2])
	x2 = layers.LSTM(256, stateful=True)(x1)
	x1 = layers.Add()([x1,x2])
	x = layers.LSTM(256, stateful=True)(x1)
#	y = layers.LSTM(inSize, stateful=True)(flatInput)
#	x = layers.concatenate([x,y])
	x = layers.Dense(lstmInSize)(x)
	# extract current time input
	forward = layers.Reshape((lowFreqRes))(inputs)
#	forward = layers.Lambda(lambda x : x[:,-1])(inputs)
	forward = tfextensions.UpsampleFreq(lowFreqRes, outputRes)(forward)
#	forward = layers.UpSampling2D(interpolation='bilinear')(forward)
	output = layers.Reshape(outputRes)(x)
	output = layers.Add()([forward, output])
	model = keras.Model(inputs=inputs, outputs=output)

	if learnFrequency:
		model.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.RMSprop(),
			metrics=['mse', tfextensions.complexMSE])
	else:
		odel.compile(loss=keras.losses.mse,
			optimizer=keras.optimizers.Adam(epsilon=0.1),
			metrics=['mse', tfextensions.frequencyLoss])

	return model

model = buildModel(batchSize)
#modelRef = tf.keras.models.load_model("vorticity3.h5")
#lstmInSize = outputRes[0]*outputRes[1]*outputRes[2]
#inSize = lowFreqRes[0] * lowFreqRes[1] * lowFreqRes[2]
#model = keras.models.Sequential([
#		layers.Reshape((windowSize,inSize), batch_input_shape=(1, windowSize,)+lowFreqRes),
#		layers.TimeDistributed(layers.Dense(inSize//4)),
#		layers.LSTM(512, activation='tanh', 
#				 stateful=True,
#				 return_sequences=True),
#		layers.LSTM(1024,
#				 return_sequences=True,
#				 stateful=True),
#		layers.LSTM(2048,
#				 stateful=True),
#	##	layers.Reshape((lowFreqRes[0],lowFreqRes[1], 4)),
#	##	layers.Conv2DTranspose(3,2,strides = 1),
#		layers.Dense(lstmInSize),
#		layers.Reshape(outputRes)
#	])
##model = buildModel(1)
#model.set_weights(modelRef.get_weights())
##model.load_weights("currentmodel/cp.ckpt")
#model.save("compress_s.h5")
#exit()


# model training
# ----------------------------------------------------------------------#
print("Starting training.")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="currentmodel/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit_generator(generateData(1024, batchSize),
							  steps_per_epoch=512, 
							  epochs=16,
							  callbacks=[cp_callback],
							  use_multiprocessing=False)

#model.save("resSimple.h5")
modelS = buildModel(1)
modelS.set_weights(model.get_weights())
modelS.save("freqRes2.h5")