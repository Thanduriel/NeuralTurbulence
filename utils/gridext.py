import numpy as np
import ctypes

FLOATP = ctypes.POINTER(ctypes.c_float)

# Returns a copy of the grid data as numpy array.
def toNumpyArray(grid, shape):
	ptrInt = int(grid.getDataPointer(), 16)
	ptr = ctypes.cast(ptrInt, FLOATP)

	arr = np.ctypeslib.as_array(ptr, shape)
	return np.copy(arr)