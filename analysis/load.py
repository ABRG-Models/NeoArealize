import numpy as np
from pathlib import Path
import h5py
import re

#
# Load and read the files in logdir.
#
def readFiles (logdir):

    # Take off any training directory slash
    logdir = logdir.rstrip ('/')

    # Read x and y first
    pf = h5py.File(logdir+'/positions.h5', 'r')
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
    p = Path(logdir+'/')
    globstr = 'c_*.h5'
    files = list(p.glob(globstr))
    # Ensure file list is in order:
    files.sort()

    numtimes = len(files)
    print ('Have {0} files/timepoints which are: {1}'.format(numtimes,files))

    # Create the time series to return. Values to be filled in from c_*.h5 file names
    t = np.empty([numtimes], dtype=int)

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
    print ('Creating empty 3d matrix of dims [{0},{1},{2}]'.format (numcs, numhexes, numtimes))
    cmatrix = np.empty([numcs, numhexes, numtimes], dtype=float)
    # There are as many 'a's as 'c's:
    amatrix = np.empty([numcs, numhexes, numtimes], dtype=float)
    nmatrix = np.empty([numhexes, numtimes], dtype=float)

    fileidx = 0
    for filename in files:

        print ('Search {0} with RE pattern {1}'.format(filename, (logdir+'/c_(.*).h5')))

        # Get the time index from the filename with a reg. expr.
        idxsearch = re.search(logdir+'/c_(.*).h5', '{0}'.format(filename))
        thetime = int('{0}'.format(idxsearch.group(1)))
        t[fileidx] = thetime
        #print ('Time {0}: {1}'.format(fileidx, thetime))

        f = h5py.File(filename, 'r')
        klist = list(f.keys())

        for k in klist:
            #print ('Key: {0}'.format(k))
            if k[0] == 'c':
                cnum = int(k[1:])
                cmatrix[cnum,:,fileidx] = np.array(f[k])
            elif k[0] == 'a':
                anum = int(k[1:])
                amatrix[anum,:,fileidx] = np.array(f[k])
            elif k[0] == 'n':
                nmatrix[:,fileidx] = np.array(f[k])

        fileidx = fileidx + 1

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
