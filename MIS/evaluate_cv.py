import AUC_library as lib
import time
import numpy as np

start_time = time.time()
dataset = np.loadtxt("../traj.txt")
#dataset = np.loadtxt("committor_time_series.txt")
sample_idx = list(range(1,dataset.shape[0],40))
sample_size = int(dataset.shape[0]/40)
print("sample_size = ", sample_size)
ncl_list = [2,8,32,48,64,80,96,192,256,384,512,648,784,1024,1280,1536,3072,4096]#,6144,8192]
#ncl_list = [2, 8, 32, 48, 64, 80, 96, 192, 256, 384, 512, 648, 784, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192, 9216, 10240, 12288, 14336, 16384, 18432]
print("ncl_list is",ncl_list)
cv = np.average(dataset,axis=1,weights=[0.50469518, -0.39708515, -0.74744887,  0.36470129, 0.68294337]).reshape(dataset.shape[0],1)
#lib.evaluate_CV(dataset[sample_idx].reshape(sample_size,1),ncl_list)
#lib.compare_to_committor(dataset,cv)
