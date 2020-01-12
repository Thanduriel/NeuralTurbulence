import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
import frequency
sys.path.append("../utils")
import ioext
import tfextensions
from saveimg import arrayToImgFile

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('--input', default='data/vorticitySymReg2/')
parser.add_argument('--output', default='predicted.mp4')
parser.add_argument('model')
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
model = tf.keras.models.load_model(args.model, tfextensions.functionMap)
timeFrame = model.input_shape[1]
isFrequency = model.input_shape[4] == 2

print("Loading data.")
if isFrequency:
	path = "data/vorticitySym2/"
else:
	path = "data/vorticitySymReg/"
inputs = ioext.loadNPData(path + "lowres_*.npy")
inputFrames, lowRes = ioext.createTimeSeries(inputs, timeFrame)
inputFrames = inputFrames[args.begin:]#inputFrames[args.begin:,0,:,:,:]
outputs = ioext.loadNPData(path + "fullres_*.npy")
simRes = outputs[0].shape
outputs = outputs[args.begin:]

print("Found {} time frames.".format(len(inputFrames)))

if args.predict or args.computeError:
	print("Applying model.")
	out = model.predict(inputFrames)
	out = np.reshape(out, (len(out),) + simRes)

outputFrames = np.reshape(outputs, (len(outputs), ) + simRes);
print("Creating images.")

if isFrequency:
	transformFn = frequency.invTransformReal
else:
	transformFn = lambda x : x[:,:,0]

if args.predict:
	for i in range(len(out)):
		vorticity = transformFn(out[i])
		arrayToImgFile(vorticity.real, "temp/predicted_{0}.png".format(i))

if args.showReference: 
	for i in range(len(outputFrames)):
	#	vorticity = inputs[i]
		vorticity = transformFn(outputFrames[i])
		arrayToImgFile(vorticity, "temp/original_{0}.png".format(i))

if args.showLowFreq: 
	highRes = np.zeros(outputFrames[0].shape)
	lowResSize = np.array(inputs[0].shape)
	begin = (outputFrames[0].shape[0] - lowResSize[0]) // 2
	end = begin + lowResSize[0]
	for i in range(len(inputs)):
		highRes[begin:end,0:lowResSize[1]] = inputs[i]
		vorticity = frequency.invTransformReal(highRes)
		arrayToImgFile(vorticity.real, "temp/lowfreq_{0}.png".format(i))

print("Creating video.")
if args.showReference:
	subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png original{}.mp4".format(args.model))
if args.predict:
	subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png -vf format=yuv420p predicted{}.mp4".format(args.model))
if args.showLowFreq:
	subprocess.run("ffmpeg -framerate 24 -i temp/lowfreq_%0d.png -vf format=yuv420p lowfreq{}.mp4".format(args.model))

def mse(array1, array2):
	dif = np.subtract(array1, array2)
	return np.mean(np.square(dif))#np.linalg.norm(dif.flatten(), ord=2)

def complexError(array1, array2):
	dif = np.subtract(frequency.flattenComplex(array1), frequency.flattenComplex(array2))

	return np.mean(np.abs(dif))

if isFrequency:
	invTransformFn = lambda x : x
else:
	invTransformFn = lambda x : frequency.stackComplex(np.fft.fftshift(np.fft.rfftn(x), axes=0))


if args.computeError:
	totalMSE = 0
	totalF = 0
	for i in range(len(out)):
		totalMSE += mse(transformFn(out[i]), transformFn(outputFrames[i]))
		totalF += complexError(invTransformFn(out[i]), invTransformFn(outputFrames[i]))
	print("mse: {}.".format(totalMSE / len(out)))
	print("freq: {}.".format(totalF / len(out)))

print("Done.")