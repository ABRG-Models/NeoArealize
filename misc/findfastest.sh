#!/bin/bash

# Find out how many threads will be fastest, up to the number of cores on the system
NUMCORES=`nproc --all`
CONFIGFILE='configs/c1.json'
for i in $(seq 1 $NUMCORES); do
    echo -n "$i cores:"
    export OMP_NUM_THREADS=${i}
    time ./build/sim/james1c ${CONFIGFILE} >/dev/null
    echo ""
done
