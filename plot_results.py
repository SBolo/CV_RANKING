import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['axes.labelsize']  = 18
rcParams['axes.titlesize']  = 18
rcParams['legend.fontsize'] = 16

def quick_load(file):
    dataset = pd.read_csv(file, delimiter=r"\s+", header = None, comment='#')
    return dataset.values

def plot_2dfes(traj, bins=80):
    # compute fes
    h, _, _ = np.histogram2d(traj[:,0], traj[:,1], bins=bins)
    fes = -np.log(h)
    fes -= np.min(fes) #shift to 0
    fes = fes.T #transpose for plotting

    # plot fes
    fig = plt.figure(figsize = (10,8))
    extent = (traj[:,0].min(), traj[:,0].max(), traj[:,1].min(), traj[:,1].max())
    contourf = plt.contourf(fes, cmap='coolwarm', zorder=-1, origin='lower', extent=extent)
    plt.colorbar()
    contour = plt.contour(fes, colors='black', origin='lower', extent=extent)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def plot_cv_fes(traj, idx, label, stride=1, dt=1, bins=80):
    # compute fes
    h = np.histogram(traj[:,idx], bins=bins)
    fes = -np.log(h[0])
    fes -= np.min(fes) #shift to 0

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(nrows=1, ncols=3)

    ax1 = fig.add_subplot(gs[0, :2])
    x = np.linspace(0, len(traj[:,idx])*stride*dt, len(traj[:,idx]))
    ax1.plot(x, traj[:,idx], c='k', lw=2)
    ax1.set_ylabel(label)
    ax1.set_xlabel('Time [a. u.]')

    ax2 = fig.add_subplot(gs[0, 2])
    plt.plot(fes, h[1][1:], c='k', lw=2)
    plt.xlabel('Free Energy [kT]')
    ax2.yaxis.set_major_formatter(plt.NullFormatter())

    fig.tight_layout()
    plt.show()

traj = quick_load('traj.txt')
plot_2dfes(traj)
#plot_cv_fes(traj, 0, 'X', bins=50, stride=100, dt=0.02)
