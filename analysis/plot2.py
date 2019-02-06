import numpy as np
# Import data loading code
import load as ld
# Import plotting code
import plot as pt
import matplotlib
import matplotlib.pyplot as plt

# Read the data
(x, y, t, cmatrix, amatrix, nmatrix) = ld.readFiles ('../logs/2N1M')

tcount = 0
for fi in range(0,len(t)):
    print ('t[fi]: {0}'.format(t[fi]))
    pt.surfaces (cmatrix[:,:,fi], amatrix[:,:,fi], nmatrix[:,fi], x, y, 'All variables')
    #path = './surfaces/surfaces{:05d}.jpg'.format(t[fi])
    # or
    path = './surfaces/surfaces{:05d}.jpg'.format(tcount)
    tcount = tcount + 1
    plt.savefig(path)
    plt.close('all')
