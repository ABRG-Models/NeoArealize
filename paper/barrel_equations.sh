#!/bin/bash

# Use convert to snip out bits of the rendered latex into png images
# suitable for Google docs

convert -verbose -density 450 -trim "barrel_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x725+390+1275 "WBEqns_01_to_03.jpg" &

convert -verbose -density 450 -trim "barrel_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+2240 "WBEqns_04_to_04.jpg" &

convert -verbose -density 450 -trim "barrel_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+2700 "WBEqns_05_to_05.jpg" &

convert -verbose -density 450 -trim "barrel_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+3250 "WBEqns_06_to_06.jpg" &

wait
