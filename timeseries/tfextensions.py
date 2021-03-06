import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def frequencyLoss(y_pred, y_true):
	freqPred = tf.signal.rfft2d(y_pred)
	freqTrue = tf.signal.rfft2d(y_true)

	return tf.math.reduce_mean(tf.math.abs(tf.math.subtract(freqPred, freqTrue)))

def complexMSE(y_pred, y_true):
	dif = tf.math.squared_difference(y_pred, y_true)
	dif = tf.math.reduce_sum(dif,axis=2)

	return tf.math.reduce_mean(dif)

class UpsampleFreq(layers.Layer):

	def __init__(self, inputRes, outputRes, **kwargs):
		super(UpsampleFreq, self).__init__(**kwargs)
		self.inputRes = inputRes
		self.outputRes = outputRes
		begin = (outputRes[0] - inputRes[0]) // 2
		end = begin + inputRes[0]
		self.paddings = tf.constant([[0, 0], # batch size 
							   [begin, outputRes[0]-end], 
							   [0, outputRes[1]-inputRes[1]],
							   [0, 0]]) # channels

	def get_config(self):
		config = super(UpsampleFreq, self).get_config()
		config.update({'inputRes': self.inputRes,
				 'outputRes': self.outputRes})

		return config



	def call(self, inputs):
		return tf.pad(inputs, self.paddings)

class OrderFreqs(layers.Layer):

	def __init__(self, inputRes, outputRes, **kwargs):
		super(OrderFreqs, self).__init__(**kwargs)
		self.inputRes = inputRes
		self.outputRes = outputRes
		self.lowFreqs = tf.zeros(outputRes.shape)
		begin = (outputRes[0] - inputRes[0]) // 2
		end = begin + inputRes[0]
		indsHigh = []
		indsLow = []
		ind = 0
		for x in range(outputRes[0]):
			for y in range(outputRes[1]):
				ind += 1
				if x >= begin and x < end and y < inputRes[1]:
					indsLow.append(ind)
				else:
				   indsHigh.append(ind)


	def get_config(self):
		config = super(OrderFreqs, self).get_config()
		config.update({'inputRes': self.inputRes,
				 'outputRes': self.outputRes})

		return config



	def call(self, inputs):
		return tf.dynamic_stitch(indices, inputs)


functionMap = {'frequencyLoss' : frequencyLoss,
			   'complexMSE' : complexMSE,
			   'UpsampleFreq' : UpsampleFreq}