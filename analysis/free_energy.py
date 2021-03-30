import numpy as np
from random import randint
import matplotlib.pyplot as plt
import os

def setup( dirr, dir_out, all_file ):

	os.system('mkdir ' + dir_out )

	for i in os.listdir(dirr):
		if( i.endswith('_folded.txt') ): #only folded trajectories are considered
			num = i.rstrip('_folded.txt')
			traj = np.loadtxt( dirr + i )
			info = 1. - np.loadtxt( dirr + 'info_' + num + '.txt' )[:,0] #1-s is the first column

			s = np.empty((len(traj),3))
			s[:,0] = traj[:,0]
			s[:,1] = traj[:,1]
			s[:,2] = info

			np.savetxt( dir_out + 's_' + num + '.txt', s, fmt = '%g' )

	os.system('cat ' + dir_out + '*.txt > ' + all_file)

def choose_ics( all_file, bins, txb ):

	trajs = np.loadtxt(all_file)
	tot = len(trajs)
	dim = (int)( (1.*tot)/2. )
	db = 1./bins
	ics = [ ] #store here the initial conditions

	b = 0
	while( b < 1. ):
		for tx in range(txb): #repeat this operation txv times for each bin
			rng = randint(0,dim)
			for i in range(rng, tot):
				t = trajs[i,2]
				if( t > b and t < b + db ):
					ics.append( [trajs[i,0], trajs[i,1]] )
					break
		b += db

	return np.array(ics)

def run( ics ):

	counter = 0
	for ic in ics:
		os.system('./langevin ' + str(ic[0]) + ' ' + str(ic[1]) + ' ' + str(counter) )
		counter += 1

def process( dirr, out_new, proc ):

	os.system('mkdir ' + out_new )

	for i in os.listdir(dirr):
		if( i.startswith('info') ): #only folded trajectories are considered
			num = i.lstrip('info_').rstrip('.txt')
			if( proc == 'preproc'):
				if os.path.isfile(dirr + num + '_folded.txt'):
					info = 1. - np.loadtxt( dirr + i )[:,0] #1-s is the first column
					np.savetxt( out_new + 's_' + num + '.txt', info, fmt = '%g' )
			elif( proc == 'postproc' ):
				info = 1. - np.loadtxt( dirr + i )[:,0] #1-s is the first column
				np.savetxt( out_new + 's_' + num + '.txt', info, fmt = '%g' )


dirr = '/media/simone/HARD_DISK/paper_giacomo/SCPS2_trajs/'
dir_out = 's/'
all_file = 'all.txt'
bins = 50
db = 1./bins #1 is the maximum value of sigma!!
txb = 2

print("Preprocess")
process(dirr, 's_SCPS/', 'preproc')
setup( dirr, dir_out, all_file )

print("Choosing initial conditions")
ics = choose_ics( all_file, bins, txb )
print(len(ics))

plt.plot(ics[:,0], ics[:,1], 'ko', markersize=1.5)
plt.show()

run(ics)

print("Postprocess")
process('trajs/', 's_MD/', 'postproc')
