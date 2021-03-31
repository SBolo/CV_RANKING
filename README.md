## Generate trajectory
1. Compile with `make`
2. Run `langevin`

## Committor
The committor was pre-computed and taken (in a 100x100 matrix form) from a previous project of mine. Details on the calculation in [1] (SI).

## Analysis
Multiple analyses can be made with `plot_results.py`:
1. plot the 2d sampled free energy (=potential in this case);
2. plot the original potential, discretized on a grid;
3. plot the free energy corresponding to a specific CV. Some are precomputed: `traj[:,0]=x`, `traj[:,1]=y`, `traj[:,2]`= distance from leftmost minimum, `traj[:,3]`=distance from central minimum, `traj[:,4]`=distance from rightmost minimum;
4. plot the probability distribution of the committor.

## References
[1] G. Bartolucci, S. Orioli, and P. Faccioli, "Transition path theory from biased simulations", J. Chem. Phys. 149, 072336 (2018). 
