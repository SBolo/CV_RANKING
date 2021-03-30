import numpy as np
import matplotlib.pyplot as plt

traj = np.loadtxt('trajs/info_1.txt')[:,0]
step = 50
stride = 70
delta_tau = 1000
length = len(traj)
rng = np.arange(0,1,1./50.)
N = []

plt.hist(traj, bins=50)
plt.show()

for i in np.arange( step, length + step, step ):

    t1 = traj[0:i]
    t2 = traj[0:i+delta_tau]

    hist1, _ = np.histogram(t1, bins = 50, normed = True )
    hist2, _ = np.histogram(t2, bins = 50, normed = True )

    hist1 = hist1/len(t1)
    hist2 = hist2/len(t2)

    dot = np.dot( hist1 - hist2, hist1 - hist2 )
    N.append(dot)

plt.plot( np.arange( step, length + step, step) * stride, np.array(N))
plt.xlabel('Time [arbitrary units]')
plt.ylabel('L2-norm')
plt.show()
