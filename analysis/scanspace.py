#
# Scan parameter space for a system with two thalamocortical axon
# types and zero guidance molecules. The idea here is to find out
# whether
#

import numpy as np
import json
import os

# Choose how many simulations to run in parallel. On corebeast, 3
# concurrent might be sensible, using 6 cores each.
parsims = 3

# Output your data here (will append)
outputfile = 'logs/scanspace.h5'

if os.path.isfile(outputfile):
    print ('\nWARNING: output.h5 ALREADY EXISTS.')
    print ()
    resp = input('Press any key to continue and append new data to {0}\n\nOR\n\nPress Ctrl-C to cancel\n'.format(outputfile))

# Scanning parameter space. Choose your params to vary and choose your
# range. The results in outputfile should be able to be combined. The
# trick with running sims in parallel will be concurrent access to the
# h5 data file.
k =     np.linspace (2, 4, 20)
alpha = np.linspace (2, 5, 20)
beta =  np.linspace (2, 5, 20)

# Estimate time, report, and allow user to bale at this point.
print ('\nASSUMING 3 SECONDS PER SIM:\nThis will take {0} hours to run'.format((len(k)*len(alpha)*len(beta)*3.0)/3600))
resp = input('Press any key to start...')

# Set unvarying parameters
simtime = 5000
jdata = {}
jdata['D'] = 0.1
jdata['steps'] = simtime
jdata['logevery'] = simtime # Only want to log the last frame
jdata['overwrite_logs'] = True
jdata['svgpath'] = './ellipse.svg'
jdata['boundaryFalloffDist'] = 0.01
jdata['guidance'] = []

configfile = 'configs/scanspace.json'

scannum = int(0)

import subprocess
import sos

for k_ in k:
    jdata['k'] = k_
    for alpha_ in alpha:
        for beta_ in beta:
            # Create temporary config json file and write out into it
            tc0 = {}
            tc0['alpha'] = alpha_
            tc0['beta'] = beta_
            jdata['tc'] = [tc0, tc0]

            jdata['logpath'] = "logs/scan"+str(scannum)

            with open(configfile, "w") as jf:
                jf.write(json.dumps(jdata))

            # Launch simulation (about 3 seconds of processing)
            subprocess.run(['./build/sim/james1c', configfile])

            # analyse and append result to h5 file
            sos.sos_centroid_analyse (jdata['logpath'], int(simtime), outputfile)

            # Done
            scannum += 1
