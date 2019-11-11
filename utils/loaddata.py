import sys
sys.path.append("../../tools")
import uniio
import os
import glob
import numpy as np

def loadData(path):
	files = []
	for entry in glob.glob(path, recursive=True):
		files.append(entry)
	# order is arbitary
	files.sort()

	data = []
	for f in files:
		header, content = uniio.readUni(f)
		h = header['dimX']
		w  = header['dimY']
		bytes = header['bytesPerElement']
		arr = content[:, ::-1, :, :] # reverse order of Y axis
		arr = np.reshape(arr, [w, h, bytes // 4])
		data.append( arr )
	return data