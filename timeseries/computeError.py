import numpy as np
import sys
sys.path.append("../utils")
import ioext 

# Computes the L2 error of the given numpy arrays.
def error(array1, array2):
	dif = np.subtract(array1, array2)
	return np.linalg.norm(dif.flatten(), ord=2)

densities = ioext.loadData("data/validation32New/density_*.uni")

total = 0
for i in range(0, len(densities)-1):
	total += error(densities[i], densities[i+1])

print(total)