import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
sys.path.append("../utils")
import ioext
from saveimg import arrayToImgFile

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('--input', default='data/validation32New/')
parser.add_argument('--output', default='predicted.mp4')
parser.add_argument('model', default='model3.h5')
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--fullpredict', dest='fullpredict', action='store_true')
parser.add_argument('--reference', dest='showReference', action='store_true')
parser.add_argument('--no-predict', dest='predict', action='store_false')
parser.add_argument('--compute-error', dest='computeError', action='store_true')
parser.set_defaults(fullpredict=False)
parser.set_defaults(showReference=False)
parser.set_defaults(predict=True)
parser.set_defaults(computeError=False)

args = parser.parse_args()
predict = args.predict or args.fullpredict
postfix = "Full" if args.fullpredict else ""

# load model now to read the size config
print("Loading model.")
model = tf.keras.models.load_model(args.model)
timeFrame = 4#model.input_shape[1]

print("Loading data.")

inputFrames, outputFrames, stepsPerSim, simRes = ioext.loadTimeseries(args.input + "density_*.uni", args.input + "vel_*.uni", timeFrame)
inputFrames = inputFrames[args.begin:]
outputFrames = outputFrames[args.begin:]
print("Found {} time frames.".format(len(inputFrames)))

print("Applying model.")
if(args.fullpredict):
	out = []
	inputFrame = inputFrames[0:1]
	for i in range(len(inputFrames)):
		output = model.predict(inputFrame)
		inputFrame[:,0:timeFrame-1] = inputFrame[:,1:]
		inputFrame[:,timeFrame-1] = np.reshape(output, simRes)
		out.append(output)
elif (args.predict or args.computeError):
	out = model.predict(inputFrames)

if predict or args.computeError:
	out = np.reshape(out, (len(out),) + simRes)
	out = out[:,:,:,0]

outputFrames = np.reshape(outputFrames, (len(outputFrames), ) + simRes);
outputFrames = outputFrames[:,:,:,0]
print("Creating images.")

if predict:
	for i in range(len(out)):
		arrayToImgFile(out[i], "temp/predicted_{0}.png".format(i))

if args.showReference: 
	for i in range(len(outputFrames)):
		arrayToImgFile(outputFrames[i], "temp/original_{0}.png".format(i))

print("Creating video.")
if args.showReference:
	subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png -vf scale=64:64 original{}.mp4".format(args.model))
if predict:
	subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png -vf scale=64:64 predicted{}{}.mp4".format(args.model,postfix))

def error(array1, array2):
	dif = np.subtract(array1, array2)
	return np.linalg.norm(dif.flatten(), ord=2)

if args.computeError:
	total = 0
	for i in range(len(out)):
		print(error(out[i], outputFrames[i]))
		total += error(out[i], outputFrames[i])
	print("Total loss: {}.".format(total))

print("Done.")