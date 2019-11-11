import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
sys.path.append("../utils")
from loaddata import loadData
from saveimg import arrayToImgFile

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('--input', default='../data/simSimple_1002/')
parser.add_argument('--output', default='predicted.mp4')
parser.add_argument('--model', default='mymodel.h5')

args = parser.parse_args()

print("Loading data.")
densities = loadData(args.input + "density_*.uni")
velocities = loadData(args.input + "vel_*.uni")
numSteps = len(densities)
simRes = densities[0].shape[0]


densities = np.reshape( densities, (len(densities),) + densities[0].shape )
velocities = np.reshape( velocities, (len(velocities),) + velocities[0].shape)

# load model now to read the size config
print("Loading model.")
model = tf.keras.models.load_model(args.model)
timeFrame = model.input_shape[1]


# prepare data
inputFrames = []
outputFrames = []
data = np.concatenate((densities,velocities), axis=3)
for i in range(0, numSteps - timeFrame):
	input = []
	for j in range(0, timeFrame):
		input.append(data[i+j].flatten())
	inputFrames.append(input)
	outputFrames.append(data[i+timeFrame])
flatSize = simRes * simRes * 4
inputFrames = np.reshape(inputFrames, (len(inputFrames),timeFrame,flatSize))
outputFrames = np.reshape(outputFrames, (len(outputFrames),flatSize))


print("Applying model.")
out = model.predict(inputFrames)
out = np.reshape(out, (len(out), simRes,simRes,4))
out = out[:,:,:,1]

outputFrames = np.reshape(outputFrames, (len(outputFrames), simRes,simRes,4));
outputFrames = outputFrames[:,:,:,1]
print("Creating images.")
for i in range(len(out)):
	arrayToImgFile(out[i], "temp/predicted_{0}.png".format(i))
	arrayToImgFile(outputFrames[i], "temp/original_{0}.png".format(i))

print("Creating video.")
subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png original.mp4")
subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png predicted.mp4")

print("Done.")