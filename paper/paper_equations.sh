#!/bin/bash

# Use convert to snip out bits of the rendered latex into png images
# suitable for Google docs

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x655+390+1275 "Eqns_01_to_03.jpg" &

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+1995 "Eqns_04_to_04.jpg" &

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+2420 "Eqns_05_to_05.jpg" &

wait
