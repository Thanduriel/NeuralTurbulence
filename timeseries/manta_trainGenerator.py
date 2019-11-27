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
sys.path.append("../utils")
import gridext

# Main params
# ----------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="Train and generate data simultaneously.")
parser.add_argument('--steps', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resolution', type=int, default=32)
parser.add_argument('--modelName', default="fastForward.h5")
parser.add_argument('--gui', dest='showGui', action='store_true')
parser.set_defaults(showGui=False)

args = parser.parse_args()

steps = args.steps
simName = "trololo"
npSeed = args.seed
resolution = args.resolution

# enable for debugging
#steps = 30 # shorter test
#savedata = False # debug , dont write...
#showGui = True

# Scene settings
# ---------------------------------------------------------------------#
setDebugLevel(0)

# NN params
# ----------------------------------------------------------------------#
windowSize = 4
batchSize = 8
lstmSize = resolution * resolution

# Solver params
# ----------------------------------------------------------------------#
res = resolution
dim = 2 
offset = 20
interval = 1

scaleFactor = 4

simRes = (res,res,1)
gs = vec3(res,res, 1 if dim == 2 else res)
buoy = vec3(0,-1e-3,0)

sm = Solver(name='smaller', gridSize = gs, dim=dim)
sm.timestep = 0.5

timings = Timings()

# Simulation Grids
# -------------------------------------------------------------------#
flags = sm.create(FlagGrid)
vel = sm.create(MACGrid)
density = sm.create(RealGrid)
pressure = sm.create(RealGrid)

# open boundaries
bWidth = 1
flags.initDomain(boundaryWidth=bWidth)
obstacle = sm.create(Sphere, center=gs*vec3(0.5,0.4,0.5), radius=res*0.10)
phiObs = obstacle.computeLevelset()
setObstacleFlags(flags=flags, phiObs=phiObs)
flags.fillGrid()
setOpenBound(flags,	bWidth,'yY',FlagOutflow | FlagEmpty) 

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
	slidingWindow = np.zeros((windowSize,)+simRes)
	t = 0
	currentInputBatch = []
	currentOutputBatch = []
	# main loop
	# --------------------------------------------------------------------#
	while 1:
		curt = t * sm.timestep
		mantaMsg("Current time t: " + str(curt) + " \n")
	
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True, boundaryWidth=bWidth)
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,	 order=2, openBounds=True, boundaryWidth=bWidth)

		if (sm.timeTotal>=0): #and sm.timeTotal<offset):
			densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
		#	sourceVel.applyToGrid( grid=vel , value=(velInflow*float(res)) )

		resetOutflow( flags=flags, real=density )
		vorticityConfinement(vel=vel, flags=flags, strength=0.05)
		setWallBcs(flags=flags, vel=vel)
		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)
	#		if (t < offset): 
	#			vorticityConfinement(vel=vel, flags=flags, strength=0.05)
		solvePressure(flags=flags, vel=vel, pressure=pressure ,  cgMaxIterFac=10.0, cgAccuracy=0.0001)

		currentVal = gridext.toNumpyArray(density,simRes)
		currentVal = currentVal[::-1,:,:];
		if t > offset:
			currentInputBatch.append(currentVal)#np.copy(slidingWindow));
			currentOutputBatch.append(currentVal)

			if(len(currentInputBatch) == batchSize):
				input = np.reshape(currentInputBatch, (batchSize,)+currentVal.shape)
				output = np.reshape(currentOutputBatch, (batchSize,)+currentVal.shape)
				yield (input, output)
				currentInputBatch = []
				currentOutputBatch = []
		
		# move window
		slidingWindow[0:windowSize-1] = slidingWindow[1:]
		slidingWindow[-1] = currentVal

		sm.step()
		t = t + 1

# model setup
# ----------------------------------------------------------------------#
lstmInSize = simRes[0]*simRes[1]*simRes[2]
model = keras.models.Sequential([
	layers.Flatten(input_shape=simRes),
	layers.Dense(lstmInSize),
#	layers.Reshape((windowSize,lstmInSize), input_shape=(windowSize,)+simRes),
#	layers.LSTM(lstmSize, activation='relu', stateful=False), # default tanh throws error "Skipping optimization due to error while loading"
	layers.Reshape(simRes)
])

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop())

#model.load_weights("models/cp.ckpt")
#model.save("model32FastForward.h5")
#exit()

# model training
# ----------------------------------------------------------------------#
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="models/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit_generator(generateData(512, 1),
							  steps_per_epoch=512, 
							  epochs=8,
							  callbacks=[cp_callback])

model.save(args.modelName)