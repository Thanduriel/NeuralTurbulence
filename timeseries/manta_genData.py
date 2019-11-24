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
sys.path.append("../tools")

# Main params
# ----------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="Generate flow data.")
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--outputPath', default='data/')
parser.add_argument('--steps', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--savedata', dest='savedata', action='store_true')
parser.add_argument('--no-savedata', dest='savedata', action='store_false')
parser.set_defaults(savedata=True)
parser.add_argument('--resolution', type=int, default=64)
parser.add_argument('--gui', dest='gui', action='store_true')
parser.set_defaults(gui=False)
parser.add_argument('--outputName', default="simSimple_")

args = parser.parse_args()

steps = args.steps
savedata = args.savedata
saveppm = False
simNo = 1000  # start ID
showGui = args.gui and args.runs == 1 # gui does not work with multiple runs
basePath = args.outputPath
simName = args.outputName
npSeed = args.seed
numRuns = args.runs
resolution = args.resolution

# enable for debugging
#steps = 30 # shorter test
#savedata = False # debug , dont write...
#showGui = True

# Scene settings
# ---------------------------------------------------------------------#
setDebugLevel(0)

# Solver params
# ----------------------------------------------------------------------#
res = resolution
dim = 2 
offset = 20
interval = 1

scaleFactor = 4

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

# Setup UI
# ---------------------------------------------------------------------#
if (showGui and GUI):
	gui = Gui()
	gui.show()
	gui.pause()

t = 0
resetN = 20
offset = 0

if savedata:
	folderNo = simNo
	pathaddition = simName + ('%04d/' % folderNo)
	while os.path.exists(basePath + pathaddition):
		folderNo += 1
		pathaddition = simName + ('%04d/' % folderNo)

	simPath = basePath + pathaddition
	print("Using output dir '%s'" % simPath) 
	simNo = folderNo
	os.makedirs(simPath)


# main loop
# --------------------------------------------------------------------#
while t < steps + offset:
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


	# save data
	if savedata and t >= offset and (t - offset) % interval == 0:
		tf = (t - offset) / interval
		#framePath = simPath + 'frame_%04d/' % tf
		#os.makedirs(framePath)
		density.save(simPath + 'density_%04d.uni' % (tf))
		vel.save(simPath + 'vel_%04d.uni' % (tf))
#		if(saveppm):
#			projectPpmFull(density, simPath + 'density_%04d_%04d.ppm' % (simNo, tf), 0, 1.0)

	sm.step()
	t = t + 1

