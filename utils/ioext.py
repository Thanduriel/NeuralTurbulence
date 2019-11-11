import sys
sys.path.append("../../tools")
import uniio
import os
import glob
import numpy as np

# load all uni files matching the given mask
def loadData(path):
	files = []
	for entry in glob.glob(path, recursive=True):
		files.append(entry)
	# order is arbitary
	files.sort()

	data = []
	for f in files:
		header, content = uniio.readUni(f)
		h = header['dimX']
		w  = header['dimY']
		bytes = header['bytesPerElement']
		arr = content[:, ::-1, :, :] # reverse order of Y axis
		arr = np.reshape(arr, [w, h, bytes // 4])
		data.append( arr )
	return data

def loadTimeseries(densityPath, velocityPath, timeFrame):
	densities = loadData(densityPath)
	velocities = loadData(velocityPath)

	stepsPerSim = len(densities)
	seriesPerSim   = stepsPerSim - timeFrame

	# create time series only from the same simulation
	inputFrames = []
	outputFrames = []
	densities = np.reshape( densities, (len(densities), ) + densities[0].shape )
	velocities = np.reshape( velocities, (len(velocities), ) + velocities[0].shape)
	data = np.concatenate((densities,velocities), axis=3)
	for i in range(0, seriesPerSim):
		input = []
		for j in range(0, timeFrame):
			input.append(data[i+j].flatten())
		inputFrames.append(input)
		outputFrames.append(data[i+timeFrame])

	simRes = data[0].shape
	flatSize = simRes[0] * simRes[1] * simRes[2]
	inputFrames = np.reshape(inputFrames, (len(inputFrames),timeFrame,flatSize))
	outputFrames = np.reshape(outputFrames, (len(outputFrames),flatSize))

	return inputFrames, outputFrames, stepsPerSim, simRes