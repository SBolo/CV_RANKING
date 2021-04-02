from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        entropies = np.array([compute_data_entropies(clusters) for clusters in clusters_list])
        # introducing last and first element
        last_element = np.array([np.log2(sample_size),0.0]).reshape(1,2)
        first_element = np.array([0.0,0.0]).reshape(1,2)
        # concatenating
        entropies = np.concatenate((first_element,entropies,last_element))
        print(entropies)
    return entropies

def compute_curve_alternative(cv_vector,full,sample_size,**kwargs):
    Z = linkage(cv_vector, 'average')
    if full == True:
        print("TODO: implement full AUC")
    elif full == False:
        if "ncl_list" not in kwargs:
            raise Exception("if full is False ncl_list should be passed!")
        ncl_list = kwargs.get("ncl_list")
        clusters_list = [fcluster(Z, t=nclust, criterion='maxclust') for nclust in ncl_list]
        entropies = np.array([compute_data_entropies(clusters) for clusters in clusters_list])
        # introducing last and first element
        last_element = np.array([np.log2(sample_size),0.0]).reshape(1,2)
        first_element = np.array([0.0,0.0]).reshape(1,2)
        # concatenating
        entropies = np.concatenate((first_element,entropies,last_element))
        #print(entropies)
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

def perturb_weights(weights):
    p_vector = np.random.normal(0.0,scale=0.01,size=weights.shape[0])
    print("weghts perturbation is ", p_vector)
    weights = weights + p_vector
    return weights

def evaluate_CV(dataset,ncl_list,**kwargs):
    """
    function that evaluates the Collective variable chosen over the trajectory.
    it could be a single CV or a linear combination of CVs 
    """
    sample_size = dataset.shape[0]
    if dataset.shape[1] > 1:
        if "weights" not in kwargs:
            raise Exception("if dataset.shape[1] > 1 weights should be passed!")
        else:
            weights = kwargs.get("weights")
            cv = np.average(dataset,axis=1,weights=weights).reshape(sample_size,1)
            if kwargs.get("save_comm") == True:
                print("saving committor to file")
                np.savetxt("cv_committor.txt",cv)
    else:
        cv = dataset.reshape(-1,1)
    print("cv shape" , cv.shape)
    cv_curve = compute_curve_alternative(cv,sample_size=sample_size,full=False,ncl_list=ncl_list)
    print("cv curve is ", cv_curve)
    AUC = compute_AUC(cv_curve[:,0],cv_curve[:,1],sample_size=sample_size)
    print("colvar has AUC ", AUC)
    if "plot" in kwargs:
        plot = kwargs.get("plot")
    if "label" in kwargs:
        label = kwargs.get("label")
        if plot == True:
            plot_curve(cv_curve[:,0],cv_curve[:,1],label=label,show=True)

def optimize_CV(dataset,ncl_list, ncv, steps,t_zero,steps_init):
    print("TODO: implement GENERATE RANDOM CV!")
    #generate_random_CV()
    weights= np.array([0.2,0.2,0.2,0.2,0.2])
    print("dataset shape", dataset.shape)
    start_cv = np.average(dataset,axis=1,weights=weights)
    curr_cv = start_cv.copy()
    print("start_cv is ", start_cv)
    sample_size = start_cv.shape[0]
    print("sample_size is ", sample_size)
    cv_curve = compute_curve_alternative(start_cv.reshape(sample_size,1),sample_size=sample_size,full=False,ncl_list=ncl_list)
    AUC = compute_AUC(cv_curve[:,0],cv_curve[:,1],sample_size=sample_size) 
    print("start colvar w =", weights, "has AUC = ", AUC)
    # highest quantity
    highest_AUC = AUC
    highest_AUC_weights = weights.copy()
    SA_step = 0
    while SA_step < steps:
        temp = t_zero*np.exp(-SA_step/steps_init)
        new_weights = perturb_weights(weights)
        new_cv = np.average(dataset,axis=1,weights=new_weights)
        cv_curve = compute_curve_alternative(new_cv.reshape(sample_size,1),sample_size=sample_size,full=False,ncl_list=ncl_list)
        new_AUC = compute_AUC(cv_curve[:,0],cv_curve[:,1],sample_size=sample_size) 
        print("SA_step", SA_step, "new_cv w =", new_weights, "has AUC = ", new_AUC)
        SA_step += 1
        # SA 
        if (new_AUC > AUC):
            print("move accepted")
            AUC = new_AUC
            #curr_cv = new_cv.copy()
            weights = new_weights.copy()
            if AUC > highest_AUC:
                print("highest_AUC found (AUC = " ,AUC , " w = ", weights,")")
                highest_AUC = AUC
                highest_AUC_weights = weights.copy()
        else:
            r = np.random.uniform(0,1)
            p = np.exp((new_AUC - AUC)/temp)
            print("t - r - p", format(t,6),format(r,6) , format(p,6))
            if p > r:
                print("p>r, move accepted")
                AUC = new_AUC
                #curr_cv = new_cv.copy()
                weights = new_weights.copy()
            else:
                print("move rejected")
        print("current cv w =", weights, "has AUC = ", AUC)
    # end of Simulated annealing
    print("END OF SIMULATED ANNEALING: highest_AUC found (AUC = " ,highest_AUC , " w = ", highest_AUC_weights,")")

def compare_to_committor(comm, cv):
    committor = pd.read_csv("../committor.txt", delimiter=r"\s+", header = None, comment='#').values
