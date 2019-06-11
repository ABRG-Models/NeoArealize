#
# Dan Whiteley's code to generate a map from one solution from his
# machine learning code.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import h5py

# takes a set of x and y coordinates for a cortex, and outputs axon
# guidance (HL2) expression patterns for each knockout

filepath = '../logs/4N0M/positions.h5'
pf = h5py.File(filepath, 'r')
klist = list(pf.keys())
for k in klist:
    if k[0] == 'x':
        xs = np.array(pf[k]);
    elif k[0] == 'y':
        ys = np.array(pf[k]);
    elif k == 'd':
        d = pf[k][0];

# check length of lists the same
if len(xs)!=len(ys):
	print("ERROR: Number of x and y coordinates must be the same")

# coordinates need to be transformed to have same scale as the ellipse
# neural network was trained on centre(3,2.5) width 10, height 8, so
# scale there here:
#
# find the range of xs and transform to -2:8
mx = max(xs)
mn = min(xs)
spread = mx-mn
for co in range(len(xs)):
	xs[co] = (xs[co]-mx)*10/spread + 8
# same for ys to -1.5:6.5
mx = max(ys)
mn = min(ys)
spread = mx-mn
for co in range(len(ys)):
	ys[co] = (ys[co]-mx)*8/spread + 6.5


# Open an HDF5 file context to save the output data into:
with h5py.File('guidance_maps.h5', 'w') as of:

    # set up network
    sizeOfInputs = 3
    sizeOfHidden1 = 11
    sizeOfHidden2 = 16
    of.create_dataset('sizeOfInputs', data=sizeOfInputs)
    of.create_dataset('sizeOfHidden1', data=sizeOfHidden1)
    of.create_dataset('sizeOfhidden2', data=sizeOfHidden2)

    weightsIH=np.array([[6.28419,0.620023,0.810574,0.279285,2.94393,-8.6944,2.57735,5.67435,-1.16826,4.53055,-1.74462],[2.82197,5.63198,1.67458,1.48695,1.28929,4.17892,0.523961,-4.09466,2.78342,2.50162,3.10915],[-3.03625,-2.1994,-2.95317,-1.82476,-9.52907,-6.20433,-11.4695,-5.54436,-10.1792,-2.08772,2.51018]])
    weightsHH=np.array([[0.972001,-2.02515,-6.02967,3.73818,-0.455873,1.12498,1.07708,0.55641,-7.54572,-7.45108,3.12948,-0.868452,3.96908,3.28243,2.67719,1.84365],[1.45296,-2.08005,-4.96141,-0.27355,-1.61275,1.5183,1.20036,1.28799,-4.60862,0.107432,0.380039,-4.05525,-2.44326,5.31324,-3.67037,-0.654167],[0.444819,2.03679,5.70548,-2.66914,1.28992,0.629034,0.150136,0.532274,5.76453,-1.93148,-1.68132,5.68876,0.0567281,3.64299,-1.67875,-1.07603],[1.04131,3.70099,8.68439,11.0731,2.56012,0.301506,0.944233,-0.182846,5.12526,-2.37583,-0.605092,7.14884,-0.808474,1.65213,0.426653,1.20754],[0.984335,0.182226,7.06435,4.50232,1.153,0.476809,0.509187,0.287508,0.6584,8.17804,-6.42726,6.18217,4.35248,3.01401,8.09005,1.63848],[1.03493,2.34486,-6.33688,2.67582,1.1683,1.2083,0.999595,0.737538,10.9299,0.0867085,2.08562,-0.221554,1.55051,-7.10955,-1.57351,-0.686278],[0.952464,-1.3625,-0.400714,2.53542,-1.28613,-0.0460613,0.164265,0.567028,6.98295,4.07632,0.39661,3.86027,1.89776,-5.14831,2.06245,0.954715],[0.855154,1.18079,5.55505,0.363761,-0.0403856,0.511434,1.02556,0.676433,-5.28248,8.43336,-0.897951,-7.13539,-0.666095,-2.89978,-1.45533,-1.24919],[0.32979,-1.68362,-1.11178,3.13264,-0.788862,0.592385,0.417414,0.162399,10.3434,-0.690582,-1.50623,6.05739,5.69991,-6.38269,2.10902,1.06286],[0.690526,-0.279825,-0.696632,-4.55903,-0.518943,0.759876,1.28817,0.996605,-3.89062,-1.37043,2.32049,-5.5752,1.07269,-1.0983,0.362515,0.222665],[1.2447,1.59787,-1.36005,-0.350512,1.34495,0.936321,1.75465,0.570021,-0.185438,-5.5594,-0.25637,-1.5115,-2.54922,-1.79119,-1.05599,0.24014]])
    weightsHO=np.array([[1.07373,-0.99039,-1.33669,-1.49004],[-3.44181,0.172731,-4.97064,2.24358],[-16.3245,-0.490498,8.86899,9.62319],[3.2283,-13.1865,-0.465315,-3.9545],[-1.64025,-1.17092,-1.94715,2.58575],[1.05545,-0.418156,-1.1548,-1.11401],[1.32795,-1.54835,-2.14776,-1.52278],[1.40258,-0.0754994,-0.918661,-0.693242],[-11.9844,-9.70723,12.5634,-5.70526],[-4.52849,-10.7945,1.19258,-11.1579],[-0.240857,5.2503,-6.82141,-3.89393],[-0.128237,16.1186,4.35799,-11.206],[0.54677,2.82322,-7.64459,3.01362],[2.28506,8.07283,10.6931,-5.2848],[0.763658,-2.05598,-7.8517,4.11927],[0.740904,-2.27624,-2.325,0.733081]])

    of.create_dataset('weightsIH', data=weightsIH)
    of.create_dataset('weightsHH', data=weightsHH)
    of.create_dataset('weightsHO', data=weightsHO)

    hidden1 = np.zeros(sizeOfHidden1)
    hidden2 = np.zeros(sizeOfHidden2)
    for knockout in range(5):
        outputValues = np.zeros((16,len(xs)))
        for coordinate in range(len(xs)):
            inputs = np.array([ys[coordinate],xs[coordinate],1])
            hidden1 = np.dot(inputs,weightsIH)
            hidden1 = 1/(1+np.exp(-hidden1))
            if knockout>0:
                hidden1[knockout-1]=0
            hidden2 = np.dot(hidden1,weightsHH)
            hidden2 = 1/(1+np.exp(-hidden2))

            for pattern in range(sizeOfHidden2):
                outputValues[pattern,coordinate] = hidden2[pattern]

        ov = of.create_dataset('knockout{0}'.format(knockout), data=outputValues)
