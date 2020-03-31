from enum import Enum

class Format(Enum):
	SPATIAL = 1
	FREQUENCY = 2

import ioext

class Inputs(Enum):
	VORTICITY = 0
	INFLOW    = 1
	PREVVORT  = 2
	OBSTACLE  = 3
	INFLOWVEL = 4

	def asOut(self):
		return "OUT_" + self.name

# returns dictionaries for input and output
def loadDataSet(path, windowSize, useLagWindows=False):
	vortInputs = ioext.loadNPData(path + "VORTICITY_*.npy")
	inputs = { Inputs.VORTICITY.name : ioext.createInputSeries(vortInputs, windowSize, useLagWindows)}
	for e in Inputs:
		data = ioext.loadNPData("{}{}_*.npy".format(path, e.name))
		if len(data) > 0:
			inputs[e.name] = ioext.createInputSeries(data, windowSize, useLagWindows)
	outputs = { Inputs.VORTICITY.asOut() : ioext.createOutputSeries(ioext.loadNPData(path + "OUT_VORTICITY_*.npy"), windowSize, useLagWindows)}

	return inputs, outputs, vortInputs

def getDataSetName(name, scale, inpF, outpF, reducedOut, seed):
	inp = "F" if inpF else "S"
	outp = "F" if outpF else "S"
	return "{}_{}_{}_{}_{}_{}".format(name,scale,inp,outp,reducedOut,seed)