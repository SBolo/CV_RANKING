import AUC_library as lib
import time
import numpy as np

start_time = time.time()
dataset = np.loadtxt("../traj.txt")
sample_idx = list(range(1,dataset.shape[0],10))
sample_size = int(dataset.shape[0]/10)
print("sample_size = ", sample_size)
ncl_list = [2,8,32,48,64,80,96,192,256,384,512,648,784,1024,1280,1536,3072,4096,6144,8192]
print("ncl_list is",ncl_list)
lib.evaluate_CV(dataset[sample_idx,:],ncl_list,weights=[-0.05524731,-0.00371473,-0.13196105,0.56902351,0.80434351],save_comm=True)