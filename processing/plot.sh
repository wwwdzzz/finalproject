#!/usr/bin/gnuplot -persist

set terminal wxt enhanced size 800,600
set datafile separator ","

set title "derivative_data"
set xlabel "X"
set ylabel "Y"

set palette defined (0 "blue", 0.5 "green", 1 "red")

plot "derivative_data.csv" using 1:2:3 with image title "Z = f(X,Y)"
