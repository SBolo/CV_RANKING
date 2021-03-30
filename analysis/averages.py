from __future__ import division
import mdtraj as md
import numpy as np
import os

trajs = 'cut_trajs/' #directory where trajectories are stored
out_file = 'z_average.txt'

llt = [ ] #list where to store the numbers of the folded trajectories
for i in os.listdir( trajs ):
	if( i.startswith('.') or i.startswith('info') ): #ignores files whose names start with '.' or 'info'
		pass
	#elif( i.endswith('_folded.txt') ):
	else:
		#llt.append( int( i.rstrip('_folded.txt') ) ) #appends only the number corresponding to the files ending with '_folded'
		llt.append( int( i.rstrip('.txt') ) )

dz = 0.05 #step in z
max_z = 5 #to set depending on the system
bins = int( (1.*max_z)/dz )
average = np.zeros( (bins,3) ) #3 means: z, x, y
mean_count = np.zeros( bins )

for num in llt:
	print("Traj ", num)

	file = trajs + str(num) + '.txt'
	traj = np.loadtxt(file) #load the trajectory file

	file = trajs + 'info_' + str(num) + '.txt'
	Z = np.loadtxt(file) #load the z files

    #Z = Z[:,0] #z is in the first column of the info file
	x = traj[:,0] #x is in the first column of the traj file
	y = traj[:,1] #y is in the second column of the traj file

	z = 0 #we start from z = 0 and grow up to max_z increasing in dz steps
	while( z < max_z ):

		counter = (int)(z/dz) #index in the arrays

		average[counter, 0] = z #set the value of z in the array

		for j in range( len(Z) ): #for the whole length of the file
			if( Z[j] >= z and Z[j] < z + dz ): #look for values of z which live in the corresponding bin
				average[counter, 1] += x[j] #average x
				average[counter, 2] += y[j] #average y
				mean_count[counter] += 1 #sum one to the counter corresponding to this value of z

		z += dz #increase z by dz

#divide by the total number of counts
average[:,1] /= mean_count
average[:,2] /= mean_count

np.savetxt("z_average.txt", average, fmt = '%g')
