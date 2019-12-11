import numpy as np
from PIL import Image

def arrayToImgFile(obj, path):
	min = np.min(obj)
	max = np.max(obj)
	obj = (obj - min) / (max-min)
	obj = (np.clip(obj, 0.0, 1.0) * 255).astype(np.uint8)
	Image.fromarray(obj).convert("L").save(path)