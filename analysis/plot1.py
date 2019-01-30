# Import data loading code
import load as ld
# Import plotting code
import plot as pt
# To access argv:
import sys

# Get target x/y hex to show trace for and the time step to show the
# map for from the arguments:
if len(sys.argv) < 4:
    print('Provide x,y,t on cmd line please.')
    exit(1)
trgx = float(sys.argv[1])
trgy = float(sys.argv[2])
trgt = int(sys.argv[3])


# External knowledge - I know that I save the time every 100 time steps:
timejump = 1

# Read the data
(x, y, t, cmatrix, amatrix, nmatrix) = ld.readFiles (timejump)

# Select an index nearest to a target
ix = ld.selectIndex (x, y, (trgx,trgy))

#print ('x range is {0} to {1}, y range is {2} to {3}'.format(min(x),max(x),min(y),max(y)))
#print ('rx is {0} for which coords are ({1},{2})'.format(ix, x[ix], y[ix]))

# Do a plot
pt.trace (cmatrix, ix, t, 'c')
pt.trace (amatrix, ix, t, 'a')
pt.trace2 (nmatrix, ix, t, 'n')

cnum = int(0)
pt.surface (cmatrix[cnum,:,trgt], x, y, ix, 'c{0}'.format(cnum))
pt.surface (amatrix[cnum,:,trgt], x, y, ix, 'a{0}'.format(cnum))
pt.surface (nmatrix[:,trgt], x, y, ix, 'n')

import matplotlib.pyplot as plt
plt.show()
