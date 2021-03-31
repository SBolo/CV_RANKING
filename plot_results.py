import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib import rcParams
from scipy.stats import gaussian_kde

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

def potential(x, y):
    x2 = x * x
    y2 = y * y
    xm1 = (x - 1) * (x - 1)
    xp1 = (x + 1) * (x + 1)
    yc = (y - 5./3.) * (y - 5./3.)
    yu = (y - 1./3.) * (y - 1./3.)
    tmp = 5. * (np.exp(-x2 - y2) - 3. / 5. * np.exp(-x2 - yc) - np.exp(-xm1 - y2) - np.exp(-xp1 - y2)) + 1. / 5. * (
                x2 * x2 + yu * yu)
    return tmp

def plot_potential(dim):
    pot = np.empty((dim, dim))
    X = np.linspace(-1.5, 1.5, dim)
    Y = np.linspace(-1., 2., dim)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            pot[i, j] = potential(x, y)

    plt.matshow(pot.T, origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.show()

def committor_from_traj(comm, traj, xlim, ylim, stride=1, dt=1):
    committor = quick_load(comm)
    q = np.empty(len(traj))
    bins = committor.shape[0]
    xr = np.linspace(xlim[0], xlim[1], bins)
    yr = np.linspace(ylim[0], ylim[1], bins)
    for k,(x,y) in enumerate(zip(traj[:,0],traj[:,1])):
        i = int((x*bins - xlim[0])/(xlim[1] - xlim[0]))
        j = int((y*bins - ylim[0])/(ylim[1] - ylim[0]))
        q[k] = committor[i,j]

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(nrows=1, ncols=3)

    ax1 = fig.add_subplot(gs[0, :2])
    x = np.linspace(0, len(traj[:,0]) * stride * dt, len(traj[:,0]))
    ax1.plot(x, q, c='k', lw=2)
    ax1.set_ylabel('Committor')
    ax1.set_xlabel('Time [a. u.]')

    ax2 = fig.add_subplot(gs[0, 2])
    plt.hist(q, bins=50, density=True, color='k', alpha=0.5, edgecolor='k', orientation='horizontal')
    plt.xlabel('Proability Density')
    ax2.yaxis.set_major_formatter(plt.NullFormatter())

    fig.tight_layout()
    plt.show()

traj = quick_load('traj.txt')
committor_from_traj("committor.txt", traj, (-1.5,1.5), (-0.25, 2.5), stride=100, dt=0.02)

#traj = quick_load('traj.txt')
#plot_2dfes(traj)
#plot_cv_fes(traj, 0, 'X', bins=50, stride=100, dt=0.02)
