#
# Post-process a set of results in logs/ps_2N0M_nnn applying the sum
# of squares analysis and writing out ps_2N0M.h5 with the final,
# plottable data.
#
# Run this from NeoArealize base directory.
#

import numpy as np
import glob, os, re
import sos

# Output your data here (will append)
outputfile = 'logs/ps_2N0M.h5'

if os.path.isfile(outputfile):
    print ('\nWARNING: {0} ALREADY EXISTS.'.format(outputfile))
    print ()
    resp = input('Press any key to continue and append new data to {0}\n\nOR\n\nPress Ctrl-C to cancel\n'.format(outputfile))

logdir = 'logs'
simtime = 5000

#regex = re.compile("logs/ps_2N0M_.*")
regex = re.compile("logs/scan.*")
for dirpath, dirnames, filenames in os.walk(logdir):
    if regex.match(dirpath):
        print ('Analyse {0}'.format(dirpath))
        # analyse and append result to h5 file
        sos.sos_centroid_analyse (dirpath, int(simtime), outputfile)
