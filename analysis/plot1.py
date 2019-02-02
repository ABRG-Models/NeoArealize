import numpy as np
# Import data loading code
import load as ld
# Import plotting code
import plot as pt
import matplotlib
import matplotlib.pyplot as plt
# To access argv:
import sys

# Set plotting defaults
fs = 12
fnt = {'family' : 'DejaVu Sans',
       'weight' : 'regular',
       'size'   : fs}
matplotlib.rc('font', **fnt)

# Get target x/y hex to show trace for and the time step to show the
# map for from the arguments:
if len(sys.argv) < 4:
    print('Provide x,y,t on cmd line please.')
    exit(1)
trgx = float(sys.argv[1])
trgy = float(sys.argv[2])
trgt = int(sys.argv[3])


# External knowledge - I know that I save the time every 100 time steps:
timejump = 3

# Read the data
(x, y, t, cmatrix, amatrix, nmatrix) = ld.readFiles (timejump)

# Select an index nearest to a target
ix = ld.selectIndex (x, y, (trgx,trgy))

#print ('x range is {0} to {1}, y range is {2} to {3}'.format(min(x),max(x),min(y),max(y)))
print ('ix is {0} for which coords are ({1},{2})'.format(ix, x[ix], y[ix]))

# Do a plot
#pt.trace (cmatrix, ix, t, 'c')
#pt.trace (amatrix, ix, t, 'a')
#pt.trace2 (nmatrix, ix, t, 'n')
pt.trace3 (amatrix, cmatrix, nmatrix, ix, t, 'Comparative')

# Plot the combined sum of n and c, which should remain constant
F1 = plt.figure (figsize=(8,8))
f1 = F1.add_subplot(1,1,1)
# Plot sum of all connections made for all c
csum = np.sum(np.sum(cmatrix[:,:,:],1), 0)
f1.plot(t, csum)
nsum = np.sum(nmatrix,0)
f1.plot(t, nsum)
f1.plot(t, csum+nsum)
f1.legend (('sum c','sum n','sum (n+c)'))
f1.set_title ('sums');

cnum = int(1)
pt.surface (cmatrix[cnum,:,trgt], x, y, ix, 'c{0}'.format(cnum))
#pt.surface (amatrix[cnum,:,trgt], x, y, ix, 'a{0}'.format(cnum))
pt.surface (nmatrix[:,trgt], x, y, ix, 'n')
pt.surface2 (nmatrix[:,trgt], x, y, ix, 'n 2')

pt.surfaces (cmatrix[:,:,trgt], amatrix[:,:,trgt], nmatrix[:,trgt], x, y, 'All variables')

plt.show()
