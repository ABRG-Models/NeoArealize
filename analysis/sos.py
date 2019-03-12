#
# To determine the similarity or difference of two spatial
# distributions of connections, just compute the sum of squared
# differences, as well as the location of the centroid.
#

def sos_centroid_analyse (logdirname, trgt, output_h5_file=''):
    import numpy as np
    # Read the config; uses json.
    import json
    jsonfilename = logdirname+'/params.json'
    jf = open (jsonfilename, 'r')
    jo = json.loads(jf.read())
    # jo is a dict, access elements like this:
    alpha = []
    beta = []
    try:
        tc_list = jo['tc'] # jo['tc'] is a list
        print ('tc_list: {0}'.format(tc_list))
        # tc is list of dicts
        for tc in tc_list:
            #print ('alpha {0}, beta {1}'.format(tc['alpha'], tc['beta']))
            alpha.append(tc['alpha'])
            beta.append(tc['beta'])

        print('alpha {0}, beta {1}'.format(alpha, beta))
    except:
        # Perhaps only alpha and beta are written in the params
        alpha.append(jo['alpha'])
        beta.append(jo['beta'])

    D = jo['D'] # D comes out as a real number. Good.
    k = jo['k'] # key error right now.
    print ('D={0}'.format(D))

    # Read the data
    import load as ld
    (x, y, t, cmatrix, amatrix, nmatrix) = ld.readFiles (logdirname)
    tidx = np.where(t==trgt)
    tidx = tidx[0][0]
    print ('type of t is {0}, tidx for {1} is {2}'.format(type(t),trgt,tidx))
    print ('shape x: {0}'.format(np.shape (x)))

    # Compute SOS
    c0 = cmatrix[0,:,tidx]
    c1 = cmatrix[1,:,tidx]
    cdiff = c0-c1
    cdiffsq = cdiff * cdiff
    sumcdiffsq = np.sum(cdiffsq)
    print ('sum of squared differences: {0}'.format(sumcdiffsq))

    # Compute Centroid.
    r_dot_a_x = np.sum(cmatrix[:,:,tidx] * x, 1)
    r_dot_a_y = np.sum(cmatrix[:,:,tidx] * y, 1)
    sigma_a = np.sum(cmatrix[:,:,tidx], 1)
    centroid_x = np.divide (r_dot_a_x, sigma_a)
    centroid_y = np.divide (r_dot_a_y, sigma_a)
    # Make a single matrix with the rows being the x/y locations of the centroids.
    centroid = np.vstack((centroid_x,centroid_y)).T
    print ('centroid: {0}'.format(centroid))

    # Append the data to an h5 file with relevant parameters. I create
    # just one dataset which is a table of parameters and the
    # corresponding sum of the square of the differences between hexes and
    # centroid positions for c0 and c1.
    import h5py
    import os
    datarow = [k, alpha[0], beta[0], D, sumcdiffsq, centroid[0][0], centroid[0][1], centroid[1][0], centroid[1][1]]
    datarowlen = len(datarow)
    newdata = np.array(datarow)
    if len(output_h5_file) > 0:
        if os.path.isfile (output_h5_file):
            with h5py.File(output_h5_file, 'a') as f1:
                # With this line, resize the data, adding a new row (the axis=1 arg)
                f1['data'].resize((f1['data'].shape[1] + 1), axis = 1)
                # Now, on the last row, set our newdata
                f1['data'][:,f1['data'].shape[1]-1] = newdata
        else:
            # create first h5. Assumes all alpha and beta entries are
            # same. This is a convention I'll stick to.
            with h5py.File(output_h5_file, 'w') as f1:
                # Create an h5 dataset. The options allow dataset to grow.
                dset = f1.create_dataset ('data', (datarowlen,1), data=newdata, chunks=True, maxshape=(datarowlen,None))
                # Create one more data set with the headings
                f1.create_dataset ('data_headings', data='k, alpha, beta, D, sumcdiffsq, centroid0x, centroid0y, centroid1x, centroid1y')

    return newdata
