import numpy as np
import matplotlib
matplotlib.use ('TKAgg', warn=False, force=True)
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

def surface (dmatrix, x, y, ix, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    F1 = plt.figure (figsize=(8,8))
    f1 = F1.add_subplot(1,1,1)
    f1.set_title(title)
    f1.scatter (x, y, c=dmatrix, marker='h', cmap=plt.cm.hsv)
    f1.scatter (x[ix], y[ix], s=32, marker='o', color='k')
