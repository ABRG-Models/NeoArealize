import sys
# example logdirname = "../logs/2N0M"
if len(sys.argv) < 3:
    print('Provide logdirname and t on command line.')
    exit(1)
logdirname = sys.argv[1]
trgt = int(sys.argv[2])

import sos

sos.sos_centroid_analyse (logdirname, trgt, 'sos.h5')
