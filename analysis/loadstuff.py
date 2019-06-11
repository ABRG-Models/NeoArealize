import numpy as np
import h5py
import re

filepath = 'positions.h5'

# Read x and y from filepath specifying an HDF5 file. pf for "pointer
# to file"
pf = h5py.File(filepath, 'r')

# Get a list of the data keys in the file
klist = list(pf.keys())

# Iterate though the keys, extracting arrays for relevant ones:
for k in klist:
    if k[0] == 'x':
        x = np.array(pf[k]);
    elif k[0] == 'y':
        y = np.array(pf[k]);
    elif k == 'd':
        d = pf[k][0];
