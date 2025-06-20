set terminal pngcairo enhanced size 1000,800
set output 'temperature_derivative.png'
set title 'Temperature Gradient (∂u/∂x) Over Time'
set xlabel 'x'
set ylabel 'Time'
set cblabel '∂u/∂x'
set pm3d map
set palette rgbformulae 22,13,-31
splot 'derivative_data.csv' u 1:2:3
