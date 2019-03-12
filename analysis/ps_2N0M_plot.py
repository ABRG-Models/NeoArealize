#
# Plot the 3D param results from ps_2N0M.h5
#
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py

# Read in the table:
f = h5py.File('logs/ps_2N0M.h5', 'r')
klist = list(f.keys())
for k in klist:
    print ('Key: "{0}"'.format(k))
    if k == 'data':
        d = np.array(f[k]).T
    elif k == 'data_headings':
        print ('Headings: {0}'.format(f[k]))

print ('shape d = {0}'.format(np.shape(d)))
# data[:,0],data[:,1],data[:,2] gives the parameter location (k,alpha,beta)
# data[:,4] is the sum of squared diffs
norm_ssdiffs = d[:,4]/max(d[:,4])

# centroid to centroid dist is:
ccdist = np.sqrt(pow((d[:,5]-d[:,7]), 2) + pow((d[:,6]-d[:,8]), 2))
norm_ccdist = ccdist/max(ccdist)

print ('length: {0}'.format(len(norm_ccdist)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('k')
ax.set_ylabel('alpha')
ax.set_zlabel('beta')

ax.scatter (d[:,0], d[:,1], d[:,2], s=20, c=norm_ccdist)
plt.show()
