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

def surface (dmatrix, x, y, ix, title):
    fs = 12
    fnt = {'family' : 'DejaVu Sans',
           'weight' : 'regular',
           'size'   : fs}
    matplotlib.rc('font', **fnt)
    F1 = plt.figure (figsize=(12,8))
    f1 = F1.add_subplot(1,1,1)
    f1.set_title(title)
    f1.scatter (x, y, c=dmatrix, marker='h', cmap=plt.cm.hsv)
    f1.scatter (x[ix], y[ix], s=32, marker='o', color='k')
