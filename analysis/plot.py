import numpy as np
import matplotlib
matplotlib.use ('TKAgg', warn=False, force=True)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def trace (dmatrix, ix, t, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    trace0 = dmatrix[0,ix,:] # c0
    trace1 = dmatrix[1,ix,:] # c1
    F1 = plt.figure (figsize=(8,8))
    f1 = F1.add_subplot(1,1,1)
    f1.plot (t,trace0)
    f1.plot (t,trace1)
    f1.set_title(title)

def trace2 (nmatrix, ix, t, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    trace0 = nmatrix[ix,:]
    F1 = plt.figure (figsize=(8,8))
    f1 = F1.add_subplot(1,1,1)
    f1.plot (t,trace0)
    f1.set_title(title)

def trace3 (a, c, n, ix, t, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    trace0 = n[ix,:]
    trace1 = a[0,ix,:] # a0
    trace2 = a[1,ix,:] # a1
    trace3 = c[0,ix,:] # c0
    trace4 = c[1,ix,:] # c1
    F1 = plt.figure (figsize=(8,8))
    f1 = F1.add_subplot(1,1,1)
    f1.plot (t,trace0,marker='o')
    f1.plot (t,trace1,marker='o')
    f1.plot (t,trace2,marker='o')
    f1.plot (t,trace3,marker='o')
    f1.plot (t,trace4,marker='o')
    f1.legend (('n','a0','a1','c0','c1'))
    f1.set_title(title)

# Plot a 2d scalar function as a colour map
def surface (dmatrix, x, y, ix, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    F1 = plt.figure (figsize=(12,8))
    f1 = F1.add_subplot(1,1,1)
    f1.set_title(title)
    f1.scatter (x, y, c=dmatrix, marker='h', cmap=plt.cm.plasma)
    f1.scatter (x[ix], y[ix], s=32, marker='o', color='k')

# Like surface, but make it a 3d projection
def surface2 (dmatrix, x, y, ix, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    F1 = plt.figure (figsize=(12,8))
    f1 = F1.add_subplot(1,1,1, projection='3d')
    f1.set_title(title)
    f1.scatter (x, y, dmatrix, c=dmatrix, marker='h', cmap=plt.cm.plasma)
    #f1.scatter (x[ix], y[ix], s=32, marker='o', color='k')

# Plot all the surfaces (c, a, j and n) in a subplot. Save a jpeg so that we can make a movie.
def surfaces (cmatrix, amatrix, jmatrix, nmatrix, x, y, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    N = np.size(cmatrix, 0)
    print ('N={0}'.format(N))
    F1 = plt.figure (figsize=(N*6,N*8))
    for i in range(0,N):
        f1 = F1.add_subplot(3,N,1+i, projection='3d')
        f1.scatter (x, y, cmatrix[i,:], c=cmatrix[i,:], marker='h', cmap=plt.cm.plasma)
        f1.set_title('c{0}'.format(i))
        f1.set_zlim(0,1)
        f2 = F1.add_subplot(3,N,N+1+i, projection='3d')
        f2.scatter (x, y, amatrix[i,:], c=amatrix[i,:], marker='h', cmap=plt.cm.plasma)
        f2.set_title('a{0}'.format(i))
        f2.set_zlim(0,1)

    # There's only one n
    f3 = F1.add_subplot(3,N,(2*N)+1, projection='3d')
    f3.scatter (x, y, nmatrix, c=nmatrix, marker='h', cmap=plt.cm.plasma)
    #f3.scatter (x, y, nmatrix, marker='o')
    f3.set_title('n'.format(i))
    f3.set_zlim(0,1)
