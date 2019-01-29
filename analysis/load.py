import numpy as np
from pathlib import Path
import h5py
import re

def readFiles(timejump):
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
        elif k[0] == 'x':
            x = np.array(f[k]);
        elif k[0] == 'y':
            y = np.array(f[k]);
        elif k == 'd':
            d = f[k][0];

    # We're expecting the data from this file to be a matrix with c0,
    # c1 etc as cols and spatial index as rows, all relating to a
    # single time point.
    print ('Creating empty 3d matrix of dims [{0},{1},{2}]'.format (numcs, numhexes,numtimes))
    dmatrix = np.empty([numcs, numhexes, numtimes], dtype=float)

    for filename in files:

        # Get the time index from the filename with a reg. expr.
        idxsearch = re.search('../logs/c_(.*).h5', '{0}'.format(filename))
        timeidx = int(-1+int('{0}'.format(idxsearch.group(1))) / timejump)
        #print ('Time index: {0}'.format(timeidx))

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
            #print ('Key: {0}'.format(k))
            if k[0] == 'c':
                # Get the data
                cnum = int(k[1:])
                dmatrix[cnum,:,timeidx] = np.array(f[k])
                # print ('dmatrix shape: {0}'.format(np.shape(dmatrix)))

    # Create the time series to return
    t = np.linspace(timejump,(timejump*numtimes),numtimes)

    return (x, y, t, dmatrix)

#
# targ is a container of a target x,y coordinate. x and y are the
# vectors of positions of the hexes in the hexgrid. This returns the
# index to the hex which is closest to targ.
#
def selectIndex(x, y, targ):
    # Now a quick plot of a select hex by time
    # Find index nearest given x and y:
    x_targ = targ[0]#-0.11
    y_targ = targ[1]# 0.4
    rmin = 10000000
    ix = 10000000
    for idx in range(0,len(x)):
        r_ = np.sqrt((x[idx] - x_targ)*(x[idx] - x_targ) + (y[idx] - y_targ)*(y[idx] - y_targ))
        if r_ < rmin:
            rmin = r_
            ix = idx
    return ix
