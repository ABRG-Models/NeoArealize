#!/bin/bash

xelatex rd_karbowski.tex
bibtex rd_karbowski.aux
xelatex rd_karbowski.tex
xelatex rd_karbowski.tex
