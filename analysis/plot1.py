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
timejump = 100

# Read the data
(x, y, t, dmatrix) = ld.readFiles (timejump)

# Select an index nearest to a target
ix = ld.selectIndex (x, y, (trgx,trgy))

#print ('x range is {0} to {1}, y range is {2} to {3}'.format(min(x),max(x),min(y),max(y)))
#print ('rx is {0} for which coords are ({1},{2})'.format(ix, x[ix], y[ix]))

# Do a plot
pt.trace (dmatrix, ix, t)

tstep = 100
cnum = 0
pt.surface (dmatrix, x, y, trgt, ix, cnum)

import matplotlib.pyplot as plt
plt.show()
