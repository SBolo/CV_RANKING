import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

x_hist = [ ]
y_hist = [ ]

fl = 'folded_SCPS0.txt'
rmsd = np.loadtxt(fl)

meanr = np.loadtxt('rMD_trajs/mean_traj.txt')
means0 = np.loadtxt('SCPS0_strong_trajs/mean_traj.txt')
means1 = np.loadtxt('trajs/mean_traj.txt')

x = rmsd[:,0]
y = rmsd[:,1]

plt.hist2d(x, y, bins=100, norm=LogNorm() )#, range = np.array([(-2, 2), (1., 2.)]))
plt.plot(meanr[:,0], meanr[:,1], 'g')
plt.plot(means0[:,0], means0[:,1], 'm')
plt.plot(means1[:,0], means1[:,1], 'r')

plt.colorbar()
plt.savefig('SCPS0.png')
