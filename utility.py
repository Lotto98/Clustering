from sklearn.base import ClusterMixin,BaseEstimator
from sklearn.metrics.cluster import rand_score

from tqdm.notebook import tqdm

from typing import List,Type, Union

import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

def hyperparameter_tuning( model: Union[Type[ClusterMixin],Type[BaseEstimator]],
                          hyperparameter_name:str,
                          hyperparameter_values:List[float], 
                          X:pd.DataFrame,
                          y:pd.Series ) -> pd.DataFrame :    
    result=[]

    for x in tqdm(hyperparameter_values):
        
        model=model.set_params(**{hyperparameter_name:x})
    
        cluster_labels=model.fit_predict(X)

        score=rand_score(y,cluster_labels)
        
        #print(model.get_params()["bandwidth"], score)
        
        result.append([x,score])
        
    result=pd.DataFrame(result,columns=[hyperparameter_name,'rand index'])
    
    print(result["rand index"].max())
        
    fitted_model=model.set_params(**{hyperparameter_name:result["rand index"].max()}).fit(X)
    
    return result,fitted_model

def plot_clustering(X, labels, cluster_centers):

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters_)]
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        
        plt.scatter(X[my_members]["PC_1"], X[my_members]["PC_2"], marker=markers[k%len(markers)], color=col)
        plt.scatter(cluster_center[0],cluster_center[1],marker=markers[k%len(markers)], edgecolor="black", s=150, color =col)
        
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()