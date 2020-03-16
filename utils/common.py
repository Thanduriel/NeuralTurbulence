from enum import Enum

class Format(Enum):
	SPATIAL = 1
	FREQUENCY = 2

import ioext

class Inputs(Enum):
	VORTICITY = 0
	INFLOW    = 1
	PREVVORT  = 2

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