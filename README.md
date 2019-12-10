# rl-project-fall2019
RL Class Project 2019

This is the README for the project. To run the code for a single bandit with
set parameters, use main.py. Parameters may be set near the top of the file.

main_grid.py runs a grid for alpha and epsilon.

main_grid_movement.py runs a grid for users moving with different step sizes.

main_grid_stepsize.py runs a grid for different delta (dampening) parameters.

These are all based on main.py and should be intuitive. Results for all are stored in results/

Plotting can be done with the respective files:
plot.py
plot_grid.py
plot_grid_movement.py
plot_grid_stepsize.py

Simply place the path to the file (e.g.'results/' + filename) in the filename
parameter and run the script. In the Github repo, sample results are included
for testing this functionality. The actual results as obtained in the paper
are much too large to upload to Github.