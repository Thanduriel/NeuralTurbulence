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

# load all npy files matching the given mask
def loadNPData(path):
	files = []
	for entry in glob.glob(path, recursive=True):
		files.append(entry)
	# load order is arbitary
	files.sort()

	data = []
	for f in files:
		arr = np.load(f)
		data.append( arr )
	return data

def createInputSeries(inputs, windowSize, lagWindows):
	steps       = len(inputs)
	numWindows  = steps - (windowSize-1)
	inputFrames = []
	stepSize    = 1 if lagWindows else windowSize

	for i in range(0, numWindows,stepSize):
		input = []
		for j in range(0, windowSize):
			input.append(inputs[i+j].flatten())
		inputFrames.append(input)

	simRes = inputs[0].shape
	inputFrames = np.reshape(inputFrames, (len(inputFrames),windowSize)+simRes)

	return inputFrames

def createOutputSeries(outputs, windowSize, lagWindows):
	outputs = outputs[windowSize-1:]
	if (not lagWindows and windowSize != 1):
		outputs = outputs[::windowSize]

	return np.reshape(outputs, (len(outputs),) + outputs[0].shape)

def createTimeSeries(inputs, outputs, windowSize, lagWindows):
	steps       = len(inputs)
	numWindows  = steps - (windowSize-1)
	inputFrames = []
	stepSize = 1 if lagWindows else windowSize

	for i in range(0, numWindows,stepSize):
		input = []
		for j in range(0, windowSize):
			input.append(inputs[i+j].flatten())
		inputFrames.append(input)

	outputs = outputs[windowSize-1:]
	if (not lagWindows and windowSize != 1):
		outputs = outputs[::windowSize]

	simRes = inputs[0].shape
	inputFrames = np.reshape(inputFrames, (len(inputFrames),windowSize)+simRes)

	return inputFrames, outputs, simRes

def createBatches(inputs, outputs, batchSize):
	assert(len(inputs) == len(outputs))

	size = len(inputs) // batchSize
	inpBatches = []
	outBatches = []	
	for i in range(size):
		inpBatch = []
		outBatch = []
		for j in range(batchSize):
			ind = i + j * size
			inpBatch.append(inputs[ind])
			outBatch.append(outputs[ind])
		inpBatches.append(inpBatch)
		outBatches.append(outBatch)

	inpBatches = np.reshape(inpBatches, (size, batchSize) + inputs[0].shape)
	outBatches = np.reshape(outBatches, (size, batchSize) + outputs[0].shape)

	return inpBatches, outBatches


def loadTimeseries(densityPath, velocityPath, timeFrame):
	densities = loadNPData(densityPath)
	#velocities = loadData(velocityPath)
	assert(len(densities))
	#assert(len(densities) == len(velocities))

	stepsPerSim = len(densities)
	seriesPerSim   = stepsPerSim - timeFrame

	# create time series only from the same simulation
	inputFrames = []
	outputFrames = []
	densities = np.reshape( densities, (len(densities), ) + densities[0].shape )
#	velocities = np.reshape( velocities, (len(velocities), ) + velocities[0].shape)
#	data = np.concatenate((densities,velocities), axis=3)
	data = densities
	for i in range(0, seriesPerSim):
		input = []
		for j in range(0, timeFrame):
			input.append(data[i+j].flatten())
		inputFrames.append(input)
		outputFrames.append(data[i+timeFrame])

	simRes = data[0].shape
	inputFrames = np.reshape(inputFrames, (len(inputFrames),timeFrame)+simRes)
	outputFrames = np.reshape(outputFrames, (len(outputFrames),)+simRes)
#	flatSize = simRes[0] * simRes[1] * simRes[2]
#	inputFrames = np.reshape(inputFrames, (len(inputFrames),timeFrame,flatSize))
#	outputFrames = np.reshape(outputFrames, (len(outputFrames),flatSize))

	return inputFrames, outputFrames, stepsPerSim, simRes