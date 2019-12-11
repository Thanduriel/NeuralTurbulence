import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
import frequency
sys.path.append("../utils")
import ioext
from saveimg import arrayToImgFile

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('--input', default='data/vorticityValidation/')
parser.add_argument('--output', default='predicted.mp4')
parser.add_argument('model', default='model3.h5')
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
predict = args.predict

# load model now to read the size config
print("Loading model.")
model = tf.keras.models.load_model(args.model)
timeFrame = model.input_shape[1]

print("Loading data.")

inputs = ioext.loadNPData(args.input + "lowres_*.npy")
inputFrames, lowRes = ioext.createTimeSeries(inputs, timeFrame)
inputFrames = inputFrames[args.begin:]#inputFrames[args.begin:,0,:,:,:]
outputs = ioext.loadNPData(args.input + "fullres_*.npy")
simRes = outputs[0].shape
outputs = outputs[args.begin:]

print("Found {} time frames.".format(len(inputFrames)))

print("Applying model.")
out = model.predict(inputFrames)

out = np.reshape(out, (len(out),) + simRes)

outputFrames = np.reshape(outputs, (len(outputs), ) + simRes);
print("Creating images.")

if predict:
	for i in range(len(out)):
		vorticity = frequency.invTransform(frequency.flattenComplex(out[i]))
		arrayToImgFile(vorticity.real, "temp/predicted_{0}.png".format(i))

if args.showReference: 
	for i in range(len(outputFrames)):
		vorticity = frequency.invTransform(frequency.flattenComplex(outputFrames[i]))
		arrayToImgFile(vorticity.real, "temp/original_{0}.png".format(i))

if args.showLowFreq: 
	highRes = np.zeros(outputFrames[0].shape)
	lowResSize = np.array(inputs[0].shape)
	begin = ((np.array(highRes.shape) - lowResSize) / 2).astype(int)
	end = (begin + lowResSize).astype(int)
	for i in range(len(inputs)):
		highRes[begin[0]:end[0],begin[1]:end[1]] = inputs[i]
		vorticity = frequency.invTransform(frequency.flattenComplex(highRes))
		arrayToImgFile(vorticity.real, "temp/lowfreq_{0}.png".format(i))

print("Creating video.")
if args.showReference:
	subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png -vf scale=64:64 original{}.mp4".format(args.model))
if predict:
	subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png -vf scale=64:64 predicted{}.mp4".format(args.model))
if args.showLowFreq:
	subprocess.run("ffmpeg -framerate 24 -i temp/lowfreq_%0d.png -vf scale=64:64 lowfreq{}.mp4".format(args.model))

def error(array1, array2):
	dif = np.subtract(array1, array2)
	return np.linalg.norm(dif.flatten(), ord=2)

if args.computeError:
	total = 0
	for i in range(len(out)):
	#	print(error(out[i], outputFrames[i]))
		total += error(out[i], outputFrames[i])
	print("Total loss: {}.".format(total))

print("Done.")