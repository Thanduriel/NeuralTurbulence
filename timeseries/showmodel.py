import tensorflow as tf
import argparse
import tfextensions

parser = argparse.ArgumentParser(description="Loads a given tf model and prints its summary.")
parser.add_argument('model')
args = parser.parse_args()

model = tf.keras.models.load_model(args.model, tfextensions.functionMap)
model.summary()