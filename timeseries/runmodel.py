import tensorflow as tf
import numpy as np
import argparse
import sys
import subprocess
sys.path.append("../utils")
import ioext
from saveimg import arrayToImgFile

parser = argparse.ArgumentParser(description="Runs a saved model and creates a video from its output.")
parser.add_argument('--input', default='../data/simSimple_1002/')
parser.add_argument('--output', default='predicted.mp4')
parser.add_argument('--model', default='mymodel.h5')

args = parser.parse_args()

# load model now to read the size config
print("Loading model.")
model = tf.keras.models.load_model(args.model)
timeFrame = model.input_shape[1]

print("Loading data.")

inputFrames, outputFrames, stepsPerSim, simRes = ioext.loadTimeseries(args.input + "density_*.uni", args.input + "vel_*.uni", timeFrame)


print("Applying model.")
out = model.predict(inputFrames)
out = np.reshape(out, (len(out),) + simRes)
out = out[:,:,:,1]

outputFrames = np.reshape(outputFrames, (len(outputFrames), ) + simRes);
outputFrames = outputFrames[:,:,:,1]
print("Creating images.")
for i in range(len(out)):
	arrayToImgFile(out[i], "temp/predicted_{0}.png".format(i))
	arrayToImgFile(outputFrames[i], "temp/original_{0}.png".format(i))

print("Creating video.")
subprocess.run("ffmpeg -framerate 24 -i temp/original_%0d.png original.mp4")
subprocess.run("ffmpeg -framerate 24 -i temp/predicted_%0d.png predicted.mp4")

print("Done.")