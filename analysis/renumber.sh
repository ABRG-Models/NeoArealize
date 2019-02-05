#!/bin/bash

counter=0
for filename in *.jpg; do
    cnt=`printf "%05d" $counter`
    cp "$filename" "../surfaces2/surfaces${cnt}.jpg"
    counter=$((counter+1))
done
