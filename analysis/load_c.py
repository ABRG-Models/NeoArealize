import numpy as np
import matplotlib
matplotlib.use ('TKAgg', warn=False, force=True)
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import re

# From ../logs get list of c_*.h5 files
p = Path('../logs/')
globstr = 'c_*.h5'
files = list(p.glob(globstr))

d = -1

gotposnX = False
gotposnY = False
gotd = False
numtimes = len(files)

print ('Have {0} files/timepoints'.format(numtimes))

f = h5py.File(files[0], 'r')
klist = list(f.keys())
# Count up how many c files we have in each time point:
numcs = 0
numhexes = 0
for k in klist:
    if k[0] == 'c':
        numcs = numcs + 1
        numhexes = len(f[k])

# We're expecting the data from this file to be a matrix with c0,
# c1 etc as cols and spatial index as rows, all relating to a
# single time point.
print ('Creating empty 3d matrix of dims [{0},{1},{2}]'.format (numcs, numhexes,numtimes))
dmatrix = np.empty([numcs, numhexes, numtimes], dtype=float)

# External knowledge - I know that I save the time every 100 time steps:
timejump = 100

for filename in files:

    # Get the time index from the filename with a reg. expr.
    idxsearch = re.search('../logs/c_(.*).h5', '{0}'.format(filename))
    timeidx = int(-1+int('{0}'.format(idxsearch.group(1))) / timejump)
    print ('Time index: {0}'.format(timeidx))

    f = h5py.File(filename, 'r')
    klist = list(f.keys())

    # Count up how many c files we have:
    numcs = 0
    numhexes = 0
    for k in klist:
        if k[0] == 'c':
            numcs = numcs + 1
            numhexes = len(f[k])

    for k in klist:
        print ('Key: {0}'.format(k))

        if gotposnX == False and k == 'x':
            x = np.array(f[k]); gotposnX = True
        elif gotposnY == False and k == 'y':
            y = np.array(f[k]); gotposnY = True
        elif gotd == False and k == 'd':
            d = f[k][0]; gotd = True

        if k[0] == 'c':
            # Get the data
            cnum = int(k[1:])
            dmatrix[cnum,:,timeidx] = np.array(f[k])
            # print ('dmatrix shape: {0}'.format(np.shape(dmatrix)))

# Now a quick plot of a select hex by time
hextoplot = 458
trace0 = dmatrix[0,458,:] # c0
trace1 = dmatrix[1,458,:] # c1
t = np.linspace(timejump,(timejump*numtimes),numtimes)

fs = 12
fnt = {'family' : 'DejaVu Sans',
       'weight' : 'regular',
       'size'   : fs}
matplotlib.rc('font', **fnt)
F1 = plt.figure (figsize=(8,8))
f1 = F1.add_subplot(1,1,1)
f1.plot (t,trace0)
f1.plot (t,trace1)
plt.show()
