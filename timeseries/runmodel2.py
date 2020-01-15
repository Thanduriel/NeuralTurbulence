import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
import time
import frequency
from prettytable import PrettyTable
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
isFrequency = model.input_shape[-1] == 2

if isFrequency:
	transformFn = lambda inp,out : frequency.invTransformReal(frequency.composeReal(out, inp, freqResOrg))
else:
	transformFn = lambda y,x : x[:,:,0]

print("Loading data.")
if isFrequency:
	path = "data/validHighFreqs/"
else:
	path = "data/vorticitySymReg/"
inputs = ioext.loadNPData(path + "lowres_*.npy")
inputFrames, lowRes = ioext.createTimeSeries(inputs, timeFrame)
inputFrames = inputFrames[args.begin:]#inputFrames[args.begin:,0,:,:,:]
outputs = ioext.loadNPData(path + "fullres_*.npy")
simRes = outputs[0].shape
freqResOrg = (64,65,2)
outputs = outputs[args.begin:]
outputFrames = np.reshape(outputs, (len(outputs), ) + simRes);

print("Found {} time frames.".format(len(inputFrames)))

excTimes = []
mseErrors = []
freqErrors = []

for modelName in args.models:
	model = tf.keras.models.load_model(modelName, tfextensions.functionMap)

	if args.predict or args.computeError:
		print("Applying model.")
		start = time.time()
		out = model.predict(inputFrames)
		timeUsed = time.time() - start
		excTimes.append(timeUsed / len(out))
		out = np.reshape(out, (len(out),) + simRes)

	print("Creating images.")

	if args.predict:
		for i in range(len(out)):
			vorticity = transformFn(inputs[i], out[i])
			arrayToImgFile(vorticity.real, "temp/predicted_{0}.png".format(i))

	if args.showReference: 
		for i in range(len(outputFrames)):
			vorticity = transformFn(inputs[i], outputFrames[i])
			arrayToImgFile(vorticity, "temp/original_{0}.png".format(i))

	if args.showLowFreq: 
		highRes = np.zeros(freqResOrg)
		lowResSize = np.array(inputs[0].shape)
		begin = (freqResOrg[0] - lowResSize[0]) // 2
		end = begin + lowResSize[0]
		for i in range(len(inputs)):
			highRes[begin:end,0:lowResSize[1]] = inputs[i]
			vorticity = frequency.invTransformReal(highRes)
			arrayToImgFile(vorticity.real, "temp/lowfreq_{0}.png".format(i))

	print("Creating video.")
	if args.showReference:
		subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png original{}.mp4".format(modelName))
	if args.predict:
		subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png -vf format=yuv420p predicted{}.mp4".format(modelName))
	if args.showLowFreq:
		subprocess.run("ffmpeg -framerate 24 -i temp/lowfreq_%0d.png -vf format=yuv420p lowfreq{}.mp4".format(modelName))

	def mse(array1, array2):
		dif = np.subtract(array1, array2)
		return np.mean(np.square(dif))#np.linalg.norm(dif.flatten(), ord=2)

	def complexError(array1, array2):
		dif = np.subtract(frequency.flattenComplex(array1), frequency.flattenComplex(array2))

		return np.mean(np.abs(dif))

	if isFrequency:
		invTransformFn = lambda x : x[:,:,0]
	else:
		invTransformFn = lambda x : frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(x), axes=0))


	if args.computeError:
		totalMSE = 0
		totalF = 0
		for i in range(len(out)):
			totalMSE += mse(transformFn(inputs[i], out[i]), transformFn(inputs[i], outputFrames[i]))
			totalF += complexError(invTransformFn(out[i]), invTransformFn(outputFrames[i]))
		mseErrors.append(totalMSE / len(out))
		freqErrors.append(totalF / len(out))
			#print("mse: {}.".format(totalMSE / len(out)))
		#print("freq: {}.".format(totalF / len(out)))

if args.predict or args.computeError:
	table = PrettyTable()
	table.add_column("model", args.models)
	if args.computeError:
		table.add_column("mse", mseErrors)
		table.add_column("freq", freqErrors)
	table.add_column("time", excTimes)

	table.float_format = "5.4"
	print(table)

print("Done.")