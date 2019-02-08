#!/bin/bash

# Find out how many threads will be fastest, up to the number of
# (physical) cores on the system
NUMCORES=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
# Include hyperthreading cores like this:
#NUMCORES=$(nproc --all)
echo "This system has ${NUMCORES} physical cores"
CONFIGFILE='configs/c1.json'
for i in $(seq 1 $NUMCORES); do
    echo -n "$i cores:"
    export OMP_NUM_THREADS=${i}
    rm -rf logs/findfastest
    time ./build/sim/james1c ${CONFIGFILE} logs/findfastest >/dev/null
    echo ""
done
