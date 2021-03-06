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
from common import *
from saveimg import arrayToImgFile

def applyLoop(model, inputs):
	s = inputs[Inputs.PREVVORT.name][0].shape
	l = len(inputs[Inputs.PREVVORT.name])
	inp = {}
	outputs = np.zeros((l,)+s)
	outputs[0] = inputs[Inputs.PREVVORT.name][0]
	for i in range(1,l):
		for key, value in inputs.items():
			if key != Inputs.PREVVORT.name:
				inp[key] = np.reshape(value[i], (1,) + value[i].shape)
		inp[Inputs.PREVVORT.name] = np.reshape(outputs[i-1], (1,) + outputs[i].shape)
		outputs[i] = model.predict(inp)

	return outputs

def mse(array1, array2):
	dif = np.subtract(array1, array2)
	return np.mean(np.square(dif))#np.linalg.norm(dif.flatten(), ord=2)

def complexError(array1, array2):
	dif = np.subtract(frequency.flattenComplex(array1), frequency.flattenComplex(array2))
	return np.mean(np.abs(dif))

def sToF(x):
	if(x.shape[-1] == 1):
	   x = x[:,:,0]
	return frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(x), axes=0))

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('models', nargs='+')
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--reference', dest='showReference', action='store_true')
parser.add_argument('--lowfreq', dest='showLowFreq', action='store_true')
parser.add_argument('--no-predict', dest='predict', action='store_false')
parser.add_argument('--compute-error', dest='computeError', action='store_true')
parser.add_argument('--powerspectrum', dest='powerSpectrum', action='store_true')
parser.add_argument('--dataset', default='all')
parser.set_defaults(fullpredict=False)
parser.set_defaults(showReference=False)
parser.set_defaults(showLowFreq=False)
parser.set_defaults(predict=True)
parser.set_defaults(computeError=False)
parser.set_defaults(powerSpectrum=False)

args = parser.parse_args()
args.computeError = args.computeError or args.powerSpectrum

# load model now to read the size config
print("Loading model.")
model = tf.keras.models.load_model(args.models[0], tfextensions.functionMap)
vorticityInputs = model.inputs[0].shape
timeFrame = vorticityInputs[1]
inputIsFrequency = vorticityInputs[-1] == 2 or vorticityInputs[-2] == 2
outputIsFrequency = model.outputs[0].shape[-1] == 2 or model.outputs[0].shape[-2] == 2
print("Detected frequency input: {}".format(inputIsFrequency))
print("Detected frequency output: {}".format(outputIsFrequency))
# currently no automatic identification possible
hasReducedOutput = True
powerSpectrumAxis = 0

if outputIsFrequency:
	toSpatial = lambda inp,out : frequency.invTransformReal(frequency.composeReal(out, inp, freqResOrg))
	toFrequency = lambda x : x
else:
	toSpatial = lambda inp,out : out if len(out.shape) == 2 else out[:,:,0]
	toFrequency = lambda x : frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(x), axes=0))

if inputIsFrequency:
	inpToFrequency = lambda x : x
else:
	inpToFrequency = sToF

print("Loading data.")
seed = 5281034
dataNames = [getDataSetName(args.dataset, 8, inputIsFrequency, outputIsFrequency, hasReducedOutput, 5281034)]#3570583
dataNames.append(getDataSetName(args.dataset, 8, inputIsFrequency, outputIsFrequency, hasReducedOutput, 1282749))


excTimes = []
mseErrors = []
freqErrors = []

for dataName in dataNames:
	path = "data/" + dataName + "/"

	inputFrames, outputs, inputs = loadDataSet(path, timeFrame, False)
	outputFrames = outputs[Inputs.VORTICITY.asOut()]
	simRes = outputFrames[0].shape
	if simRes[-1] == 1:
		simRes = simRes[0:2]
	freqResOrg = (64,65,2)

	if outputFrames.shape[-1] == 1:
		outputFrames = np.reshape(outputFrames, outputFrames.shape[0:-1])
	#outputFrames = np.reshape(outputs, (len(outputs), simRes[0], simRes[1]));

	outputFramesSpat = []
	outputFramesFreq = []
	for i in range(len(outputFrames)):
		outputFramesSpat.append(toSpatial(inpToFrequency(inputs[i]), outputFrames[i]))
		outputFramesFreq.append(toFrequency(outputFrames[i]))

	print("Found {} time frames.".format(len(inputFrames)))

	if args.powerSpectrum:
		highRes = np.zeros(freqResOrg)
		frequencyComponents, psTotalZero = frequency.getSpectrum(highRes, powerSpectrumAxis)
		psTotalRef = np.copy(psTotalZero)
		for outp in outputFramesSpat:
			_, ps = frequency.getSpectrum(sToF(outp), powerSpectrumAxis) 
			psTotalRef += ps
		psTotalRef /= len(outputFramesFreq)
		plt.semilogy(frequencyComponents, psTotalRef)
		legendEntries = ["reference"]

	if args.showReference: 

		for i in range(len(outputFrames)):
			arrayToImgFile(outputFramesSpat[i], "temp/original_{0}.png".format(i))
		
		subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png original{}.mp4".format(dataName))

	outputFramesLow = []
	if args.showLowFreq or hasReducedOutput: 
		highRes = np.zeros(freqResOrg)
		highResReduced = np.zeros(outputFramesFreq[0].shape)
		lowResSize = np.array(inpToFrequency(inputs[0]).shape)
		begin = (freqResOrg[0] - lowResSize[0]) // 2
		end = begin + lowResSize[0]

		totalMSE = 0
		totalF = 0
		if args.powerSpectrum:
			psTotal = np.copy(psTotalZero)

		for i in range(len(inputs)):
			highRes[begin:end,0:lowResSize[1]] = inpToFrequency(inputs[i])
			vorticity = frequency.invTransformReal(highRes)
			outputFramesLow.append(vorticity)

			if args.computeError:
				totalMSE += mse(vorticity, outputFramesSpat[i])
				totalF += complexError(highResReduced, outputFramesFreq[i])
			if args.powerSpectrum:
				freqs_, ps = frequency.getSpectrum(highRes, powerSpectrumAxis)
				psTotal += ps

		if args.computeError and args.showLowFreq:
			mseErrors.append(totalMSE / len(outputFrames))
			freqErrors.append(totalF / len(outputFrames))
			excTimes.append(0.0)

		if args.powerSpectrum:
			plt.semilogy(frequencyComponents, np.abs((psTotal / len(inputs))) )#psTotalRef -
			legendEntries.append("lowfreq")

	if args.showLowFreq:
		for i in range(len(inputs)):
			arrayToImgFile(outputFramesLow[i], "temp/lowfreq_{0}.png".format(i))
		subprocess.run("ffmpeg -framerate 24 -i temp/lowfreq_%0d.png -vf format=yuv420p lowfreq{}.mp4".format(dataName))

	for modelName in args.models:
		model = tf.keras.models.load_model(modelName, tfextensions.functionMap)
		sErrors = []
		fErrors = []

		if args.predict or args.computeError:
			print("Running model {}.".format(modelName))
			start = time.time()
			if Inputs.PREVVORT.name in model.inputs[0].name:
				out = applyLoop(model, inputFrames)
			else:
				out = model.predict(inputFrames)
			timeUsed = time.time() - start
			excTimes.append(timeUsed / len(out))
			out = np.reshape(out, (len(out),) + simRes)

		if args.predict:
			for i in range(len(out)):
				vorticity = toSpatial(inpToFrequency(inputs[i]), out[i])
				if hasReducedOutput and not outputIsFrequency:
					vorticity += outputFramesLow[i]
				arrayToImgFile(vorticity.real, "temp/predicted_{0}.png".format(i))

			subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png -vf format=yuv420p predicted{}.mp4".format(modelName))

		if args.computeError:
			totalMSE = 0
			totalF = 0
			if args.powerSpectrum:
				psTotal = np.copy(psTotalZero)
			for i in range(len(out)):
				outSpat = toSpatial(inpToFrequency(inputs[i]), out[i])
				totalMSE += mse(outSpat, outputFramesSpat[i])
				outFreq = toFrequency(out[i])
				#if i  < 80:
				sErrors.append(mse(outSpat, outputFramesSpat[i]))
				fErrors.append(complexError(outFreq, outputFramesFreq[i]))
				totalF += complexError(outFreq, outputFramesFreq[i])
			
				if args.powerSpectrum:
					freqs_, ps = frequency.getSpectrum(sToF(outSpat), powerSpectrumAxis)
					psTotal += ps

			if args.powerSpectrum:
				plt.semilogy(frequencyComponents, np.abs((psTotal / len(out))) )#psTotalRef -
				legendEntries.append(modelName)
				#np.save("s_f", np.abs((psTotal / len(out))))
			mseErrors.append(totalMSE / len(out))
			freqErrors.append(totalF / len(out))
			np.save("{}_sErr".format(modelName), np.reshape(sErrors, (len(out),)))
			np.save("{}_fErr".format(modelName), np.reshape(fErrors, (len(out),)))


def avgResults(arr):
	numEntries = len(arr) // len(dataNames)
	for i in range(numEntries):
		for j in range(1,len(dataNames)):
			arr[i] += arr[i + j * numEntries]
		arr[i] = arr[i] / len(dataNames)

	return arr[0:numEntries]

if args.computeError:
	mseErrors = avgResults(mseErrors)
	freqErrors = avgResults(freqErrors)
excTimes = avgResults(excTimes)


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

if args.powerSpectrum:
	plt.legend(legendEntries)
	plt.show()

print("Done.")