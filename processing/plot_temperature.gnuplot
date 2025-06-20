set terminal pngcairo enhanced size 1000,800
set output 'temperature_evolution.png'
set title 'Temperature Evolution Over Time'
set xlabel 'x'
set ylabel 'y'
set cblabel 'Temperature'
set pm3d map
set palette rgbformulae 22,13,-31
splot 'solution_0.0000.csv' u 1:2:3 title 't=0', \
      'solution_0.2500.csv' u 1:2:3 title 't=0.25T', \
      'solution_0.5000.csv' u 1:2:3 title 't=0.5T', \
      'solution_0.7500.csv' u 1:2:3 title 't=0.75T', \
      'solution_1.0000.csv' u 1:2:3 title 't=T'
