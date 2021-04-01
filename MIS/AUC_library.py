from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_data_entropies(clusters):
    N = len(clusters)
    #print("N = ",N)
    ks = np.unique(clusters,return_counts=True)
    # let's extract the counts mk, whose first element will be equal to the frequencies. mk[0] == ks[1] 
    mk = np.unique(ks[1],return_counts=True)
    #print(ks)
    #print(mk)
    # cluster-wise resolutions
    res_vect = [-mk[1][n]*mk[0][n]/N*np.log2(mk[0][n]/N) for n in range(len(mk[0]))]
    #print(res_vect)
    # cluster-wise relevances
    rel_vect = [-mk[1][n]*mk[0][n]/N*np.log2(mk[0][n]*mk[1][n]/N) for n in range(len(mk[0]))]
    #print(rel_vect)
    return sum(res_vect), sum(rel_vect)

def compute_curve(dist_matrix, full, sample_size,**kwargs):
    Z = linkage(dist_matrix, 'average')
    if full == True:
        print("TODO: implement full AUC")
    elif full == False:
        if "ncl_list" not in kwargs:
            raise Exception("if full is False ncl_list should be passed!")
        ncl_list = kwargs.get("ncl_list")
        clusters_list = [fcluster(Z, t=nclust, criterion='maxclust') for nclust in ncl_list]
        #print(clusters_list)
        entropies = np.array([compute_data_entropies(clusters) for clusters in clusters_list])
        print(entropies.shape)
        last_element = np.array([np.log2(sample_size),0.0]).reshape(1,2)
        first_element = np.array([0.0,0.0]).reshape(1,2)
        print(last_element.shape)
        entropies = np.concatenate((first_element,entropies,last_element))
        #entropies = np.append(entropies,,axis=0)
        print(entropies)
    return entropies

def plot_curve(resolution,relevance,label,color,show):
    #plt.plot(resolution,relevance,linestyle='--', marker='o',label="CV = " + label,color=color)
    plt.plot(resolution,relevance,linestyle='--',label="CV = " + label,color=color)
    if show == True:
        plt.show()

def compute_AUC(resolution,relevance,sample_size):
    #np.append(resolution,np.log2(sample_size))
    #np.append(relevance, 0.0)
    AUC = np.trapz(relevance,resolution)
    return AUC

def retrieve_curve_points(sample_size,n_points):
    """
    This routine gives a reasonable set of points over which we can compute the data entropies instead of calculating them over the full curve
    NB: we don't need many points in the leftmost part of the curve, but we need super frequent sampling around the maximum
    NB2: you often observe the maximum of the relevance if ncl ~ 500 
    """


#ncl_list = [2,4,8,16,32,64,128,256,512,1024,2048,3072,4096,6144,7168,8192,9216]

dataset = np.loadtxt("../traj.txt")
sample_idx = list(range(1,dataset.shape[0],4))
sample_size = int(dataset.shape[0]/4)
print(sample_idx)
ncl_list = [2,8,32,64,80,96,128,192,256,320,384,448,512,768,1024,1280,1536,1792,2048,3072,4096,6144,7168,8192,9216,10240,12288,14336,16384,18432]
colvars = {0:"x", 1:"y", 2:"z1", 3:"z2", 4:"z3"}
colors = {0:"red", 1:"blue", 2:"green", 3:"lightgreen", 4:"pink"}
for n in range(dataset.shape[1]):
    print("colvar", n)
    #distances = pdist(dataset[:,n].reshape(dataset.shape[0],1),"euclidean")
    distances = pdist(dataset[sample_idx,n].reshape(25000,1),"euclidean")
    cv_curve = compute_curve(distances,sample_size=sample_size,full=False,ncl_list=ncl_list)
    plot_curve(cv_curve[:,0],cv_curve[:,1],label=colvars[n],color=colors[n],show=False)
    AUC = compute_AUC(cv_curve[:,0],cv_curve[:,1],sample_size)
    print("colvar", colvars[n], "has AUC = ", AUC)
#print(dataset.shape)
plt.legend(fontsize=12)
plt.xlabel("$H_s$",fontsize=16)
plt.ylabel("$H_k$",fontsize=16)
plt.savefig("")
plt.show()
