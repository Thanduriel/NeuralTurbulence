import tensorflow as tf
import argparse
import tfextensions

parser = argparse.ArgumentParser(description="Loads a given tf model and plots it.")
parser.add_argument('model')
args = parser.parse_args()

model = tf.keras.models.load_model(args.model, tfextensions.functionMap)
tf.keras.utils.plot_model(model, args.model+".pdf", show_shapes=True, show_layer_names=False)