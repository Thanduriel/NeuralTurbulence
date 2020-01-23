import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
import time
import frequency
from prettytable import PrettyTable
import matplotlib.pyplot as plt
sys.path.append("../utils")
import ioext
import tfextensions
from saveimg import arrayToImgFile

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('--input', default='data/vorticitySymReg2/')
parser.add_argument('--output', default='predicted.mp4')
parser.add_argument('models', nargs='+')
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--reference', dest='showReference', action='store_true')
parser.add_argument('--lowfreq', dest='showLowFreq', action='store_true')
parser.add_argument('--no-predict', dest='predict', action='store_false')
parser.add_argument('--compute-error', dest='computeError', action='store_true')
parser.set_defaults(fullpredict=False)
parser.set_defaults(showReference=False)
parser.set_defaults(showLowFreq=False)
parser.set_defaults(predict=True)
parser.set_defaults(computeError=False)

args = parser.parse_args()

# load model now to read the size config
print("Loading model.")
model = tf.keras.models.load_model(args.models[0], tfextensions.functionMap)
timeFrame = model.input_shape[1]
inputIsFrequency = model.input_shape[-1] == 2
outputIsFrequency = model.output_shape[-1] == 2
# currently no automatic identification possible
hasReducedOutput = True

if outputIsFrequency:
	toSpatial = lambda inp,out : frequency.invTransformReal(frequency.composeReal(out, inp, freqResOrg))
	toFrequency = lambda x : x
else:
	toSpatial = lambda inp,out : out if len(out.shape) == 2 else out[:,:,0]
	toFrequency = lambda x : frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(x), axes=0))

print("Loading data.")
if outputIsFrequency:
	dataName = "validHighFreqs"
else:
	dataName = "valid_4_f_sr"
path = "data/" + dataName + "/"

inputs = ioext.loadNPData(path + "lowres_*.npy")
inputFrames, lowRes = ioext.createTimeSeries(inputs, timeFrame)
inputFrames = inputFrames[args.begin:]#inputFrames[args.begin:,0,:,:,:]
outputs = ioext.loadNPData(path + "fullres_*.npy")
simRes = outputs[0].shape
freqResOrg = (64,65,2)
outputs = outputs[args.begin:]

if outputIsFrequency:
	outputFrames = np.reshape(outputs, (len(outputs), ) + simRes);
else:
	outputFrames = np.reshape(outputs, (len(outputs), simRes[0], simRes[1]));

outputFramesSpat = []
outputFramesFreq = []
for i in range(len(outputFrames)):
	outputFramesSpat.append(toSpatial(inputs[i], outputFrames[i]))
	outputFramesFreq.append(toFrequency(outputFrames[i]))

print("Found {} time frames.".format(len(inputFrames)))

excTimes = []
mseErrors = []
freqErrors = []

def mse(array1, array2):
	dif = np.subtract(array1, array2)
	return np.mean(np.square(dif))#np.linalg.norm(dif.flatten(), ord=2)

def complexError(array1, array2):
	dif = np.subtract(frequency.flattenComplex(array1), frequency.flattenComplex(array2))
	return np.mean(np.abs(dif))

if args.showReference: 

	for i in range(len(outputFrames)):
		arrayToImgFile(outputFramesSpat[i], "temp/original_{0}.png".format(i))
		
	subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png original{}.mp4".format(dataName))

outputFramesLow = []
if args.showLowFreq or hasReducedOutput: 
	highRes = np.zeros(freqResOrg)
	lowResSize = np.array(inputs[0].shape)
	begin = (freqResOrg[0] - lowResSize[0]) // 2
	end = begin + lowResSize[0]

	totalMSE = 0
	totalF = 0

	for i in range(len(inputs)):
		highRes[begin:end,0:lowResSize[1]] = inputs[i]
		vorticity = frequency.invTransformReal(highRes)
		outputFramesLow.append(vorticity)

		shrinked = frequency.shrink(highRes, lowResSize)
		totalMSE += mse(vorticity, outputFramesSpat[i])
		totalF += complexError(highRes, outputFramesFreq[i])

	mseErrors.append(totalMSE / len(outputFrames))
	freqErrors.append(totalF / len(outputFrames))
	excTimes.append(0.0)

if args.showLowFreq:
	for i in range(len(inputs)):
		arrayToImgFile(outputFramesLow[i], "temp/lowfreq_{0}.png".format(i))
	subprocess.run("ffmpeg -framerate 24 -i temp/lowfreq_%0d.png -vf format=yuv420p lowfreq{}.mp4".format(dataName))

for modelName in args.models:
	print(modelName)
	model = tf.keras.models.load_model(modelName, tfextensions.functionMap)

	if args.predict or args.computeError:
		print("Applying model.")
		start = time.time()
		out = model.predict(inputFrames)
		timeUsed = time.time() - start
		excTimes.append(timeUsed / len(out))
		out = np.reshape(out, (len(out),) + simRes)

#	highRes = np.zeros(freqResOrg)
#	freqs, psTotal = frequency.getSpectrum(highRes, 0)
	if args.predict:
		for i in range(len(out)):
			vorticity = toSpatial(inputs[i], out[i])
			if hasReducedOutput and not outputIsFrequency:
				vorticity += outputFramesLow[i]
			arrayToImgFile(vorticity.real, "temp/predicted_{0}.png".format(i))
		#	highRes = frequency.composeReal(out[i],inputs[i], freqResOrg)
		#	freqs_, ps = frequency.getSpectrum(highRes, 0)
		#	psTotal += ps
#		plt.plot(freqs, psTotal / len(inputs) )
#		plt.show()

		subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png -vf format=yuv420p predicted{}.mp4".format(modelName))

	if args.computeError:
		totalMSE = 0
		totalF = 0
		for i in range(len(out)):
			totalMSE += mse(toSpatial(inputs[i], out[i]), outputFramesSpat[i])
			totalF += complexError(toFrequency(out[i]), outputFramesFreq[i])
		mseErrors.append(totalMSE / len(out))
		freqErrors.append(totalF / len(out))
			#print("mse: {}.".format(totalMSE / len(out)))
		#print("freq: {}.".format(totalF / len(out)))


if args.predict or args.computeError:
	table = PrettyTable()
	names = args.models
	if args.showLowFreq:
		names.insert(0,"lowfreq")
	table.add_column("model", names)
	if args.computeError:
		table.add_column("mse", mseErrors)
		table.add_column("freq", freqErrors)
	table.add_column("time", excTimes)

	table.float_format = "5.4"
	print(table)

print("Done.")