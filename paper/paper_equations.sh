#!/bin/bash

# Use convert to snip out bits of the rendered latex into png images
# suitable for Google docs

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x675+390+1255 "Eqns_01_to_03.jpg" &

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+1995 "Eqns_04_to_04.jpg" &

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x675+390+2735 "Eqns_05_to_07.jpg" &

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x245+390+3465 "Eqns_08_to_08.jpg" &

convert -verbose -density 450 -trim "paper_equations.pdf[0]" -quality 100 -antialias -flatten -crop 2950x235+390+3885 "Eqns_09_to_09.jpg" &

wait
