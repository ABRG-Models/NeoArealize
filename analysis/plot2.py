import numpy as np
# Import data loading code
import load as ld
# Import plotting code
import plot as pt
import matplotlib
import matplotlib.pyplot as plt

# External knowledge - I know that I save the time every n time
# steps. Might be nice if this was stored in a file...
timejump = 20

# Read the data
(x, y, t, cmatrix, amatrix, nmatrix) = ld.readFiles (timejump)

#timepoint=int(1000)
tpcount = 0
for timepoint in t:
    tp = int(timepoint)
    pt.surfaces (cmatrix[:,:,tp], amatrix[:,:,tp], nmatrix[:,tp], x, y, 'All variables')
    path = './surfaces/surfaces{:05d}.jpg'.format(tpcount)
    tpcount = tpcount + 1
    plt.savefig(path)
    plt.close('all')
