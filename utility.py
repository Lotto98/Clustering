from sklearn.base import ClusterMixin,BaseEstimator
from sklearn.metrics.cluster import rand_score

from tqdm.notebook import tqdm

from typing import List,Type, Union, Tuple

import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

def load_PCA_dfs(n:int)->Tuple[dict,pd.Series]:
    
    y = pd.read_parquet('dataset/y.parquet').squeeze()
    
    dfs={}
    
    for i in tqdm(range(2,n+1,20)):
        
        dfs[i]=pd.read_parquet("dataset/PCA_"+str(i)+".parquet")
        
    return dfs,y
        

def hyperparameter_tuning(desc:str,
                          model: Union[Type[ClusterMixin],Type[BaseEstimator]],
                          hyperparameter_name:str,
                          hyperparameter_values:List[float], 
                          X:pd.DataFrame,
                          y:pd.Series,
                          return_fitted: bool = False )-> Tuple[ pd.DataFrame,
                                                                pd.DataFrame,
                                                                Union[Type[ClusterMixin],Type[BaseEstimator]]]:    
    result=[]
    fitted_model=None

    for x in tqdm(hyperparameter_values,desc=desc):
        
        model=model.set_params(**{hyperparameter_name:x})
    
        cluster_labels=model.fit_predict(X)

        score=rand_score(y,cluster_labels)
        
        #print(model.get_params()["bandwidth"], score)
        
        result.append([x,score])
        
    result=pd.DataFrame(result,columns=[hyperparameter_name,'rand index'])
    
    best_result_index=result.index[result["rand index"]==result["rand index"].max()].to_list()[0]
    
    if return_fitted:
        fitted_model=model.set_params(**{hyperparameter_name:result[hyperparameter_name].iloc[best_result_index].squeeze()}).fit(X)
    
    return result, best_result_index, fitted_model

def get_results(dfs:dict,
                y:pd.Series,
                estimator:Union[Type[ClusterMixin],Type[BaseEstimator]],
                hyperparameter_name:str,
                hyperparameter_values:List[float]) -> Tuple[dict,dict,Union[Type[ClusterMixin],Type[BaseEstimator]]]:
    
    results={}
    best_indexes={}

    #dfs.keys()
    
    for dim in tqdm([2],desc="Total result"):
        
        get_estimator = dim==2
        
        results[dim],best_indexes[dim],_=hyperparameter_tuning("PCA_"+str(dim),
                                                               estimator,hyperparameter_name,
                                                               hyperparameter_values,
                                                               dfs[dim], y,
                                                               get_estimator)
        
        if get_estimator:                                                               
            estimator2D=_
        
    return results, best_indexes, estimator2D

def plot_clustering(X, labels, cluster_centers=None):

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters_)]
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        
        plt.scatter(X[my_members]["PC_1"], X[my_members]["PC_2"], marker=markers[k%len(markers)], color=col)
        
        if cluster_centers:
            cluster_center = cluster_centers[k]
            plt.scatter(cluster_center[0],cluster_center[1],marker=markers[k%len(markers)], edgecolor="black", s=150, color =col)
        
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()