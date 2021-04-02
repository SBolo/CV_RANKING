import AUC_library as lib
import time
import numpy as np

start_time = time.time()
dataset = np.loadtxt("../traj.txt")
sample_idx = list(range(1,dataset.shape[0],10))
sample_size = int(dataset.shape[0]/10)
print("sample_size = ", sample_size)
#ncl_list = [2,8,32,64,80,96,128,192,256,320,384,448,512,768,1024,1280,1536,1792,2048,3072,4096,6144,7168,8192,9216,10240,12288,14336,16384,18432]
#ncl_list = [2,8,32,64,80,96,128,192,256,320,384,448,512,768,1024,1280,1536,1792,2048,3072,4096,6144,7168,8192,9216,10240,12288,14336,16384,18432,20736,23040,27648,32256,36864,41472,46080]
ncl_list = [2,8,32,48,64,80,96,192,256,384,512,648,784,1024,1280,1536,3072,4096,6144,8192]
print("ncl_list is",ncl_list)
t_zero = 1.0
steps_init = 2500
#steps = 10000
steps = 10
lib.optimize_CV(dataset[sample_idx,:],ncl_list,steps,steps,t_zero,steps_init)
