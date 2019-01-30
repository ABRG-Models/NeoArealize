import numpy as np
from pathlib import Path
import h5py
import re

def readFiles(timejump):

    # Read x and y first
    pf = h5py.File('../logs/positions.h5', 'r')
    klist = list(pf.keys())
    # Count up how many c files we have in each time point:
    d = -1
    for k in klist:
        if k[0] == 'x':
            x = np.array(pf[k]);
        elif k[0] == 'y':
            y = np.array(pf[k]);
        elif k == 'd':
            d = pf[k][0];

    # From ../logs get list of c_*.h5 files
    p = Path('../logs/')
    globstr = 'c_*.h5'
    files = list(p.glob(globstr))

    numtimes = len(files)
    print ('Have {0} files/timepoints'.format(numtimes))

    # Count up how many c files we have in each time point once only:
    f = h5py.File(files[0], 'r')
    klist = list(f.keys())
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
    cmatrix = np.empty([numcs, numhexes, numtimes], dtype=float)
    # There are as many 'a's as 'c's:
    amatrix = np.empty([numcs, numhexes, numtimes], dtype=float)
    nmatrix = np.empty([numhexes, numtimes], dtype=float)

    for filename in files:

        # Get the time index from the filename with a reg. expr.
        idxsearch = re.search('../logs/c_(.*).h5', '{0}'.format(filename))
        timeidx = int(-1+int('{0}'.format(idxsearch.group(1))) / timejump)
        #print ('Time index: {0}'.format(timeidx))

        f = h5py.File(filename, 'r')
        klist = list(f.keys())

        for k in klist:
            #print ('Key: {0}'.format(k))
            if k[0] == 'c':
                cnum = int(k[1:])
                cmatrix[cnum,:,timeidx] = np.array(f[k])
            elif k[0] == 'a':
                anum = int(k[1:])
                amatrix[anum,:,timeidx] = np.array(f[k])
            elif k[0] == 'n':
                nmatrix[:,timeidx] = np.array(f[k])

    # Create the time series to return
    t = np.linspace(timejump,(timejump*numtimes),numtimes)

    return (x, y, t, cmatrix, amatrix, nmatrix)

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
