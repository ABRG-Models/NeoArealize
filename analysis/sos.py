#
# To determine the similarity or difference of two spatial
# distributions of connections, just compute the sum of squared
# differences, as well as the location of the centroid.
#

import numpy as np
# Import data loading code
import load as ld
# Import plotting code
import plot as pt
import matplotlib
import matplotlib.pyplot as plt
# To access argv:
import sys

# Read the data
logdirname = "../logs/2N0M"
(x, y, t, cmatrix, amatrix, nmatrix) = ld.readFiles (logdirname)
trgt = 1000

tidx = np.where(t==trgt)
tidx = tidx[0][0]
print ('type of t is {0}, tidx for {1} is {2}'.format(type(t),trgt,tidx))

print ('shape x: {0}'.format(np.shape (x)))

# Compute SOS
c0 = cmatrix[0,:,tidx]
c1 = cmatrix[1,:,tidx]

cdiff = c0-c1
cdiffsq = cdiff * cdiff
sumcdiffsq = np.sum(cdiffsq)
#print ('{0}'.format (cdiff))
#print ('shape cdiffsq: {0}, sum: {1} sum c0: {2} sum c1: {3}'.format(np.shape (cdiffsq), sumcdiffsq, np.sum(c0), np.sum(c1)))
print ('sum of squared differences: {0}'.format(sumcdiffsq))

# Compute Centroid.
r_dot_a_x = np.sum(cmatrix[:,:,tidx] * x, 1)
r_dot_a_y = np.sum(cmatrix[:,:,tidx] * y, 1)
print ('r.a x components sum to {0}'.format (r_dot_a_x))
print ('r.a y components sum to {0}'.format (r_dot_a_y))

sigma_a = np.sum(cmatrix[:,:,tidx], 1)
print ('sum a {0}'.format (sigma_a))
centroid_x = np.divide (r_dot_a_x, sigma_a)
centroid_y = np.divide (r_dot_a_y, sigma_a)
# Make a single matrix with the rows being the x/y locations of the centroids.
centroid = np.vstack((centroid_x,centroid_y)).T
print ('centroid: {0}'.format(centroid))

# Now do something with the data. Write out into h5 file with relvant parameters?
