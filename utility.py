from sklearn.base import ClusterMixin,BaseEstimator
from sklearn.metrics.cluster import rand_score

from tqdm.notebook import tqdm

from typing import List,Type, Union, Tuple

import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

import copy

import pickle

def save_results(model_name:str,results:dict,best_indexes:dict,fitted_models:dict):
    
    with open(model_name+"/results.pkl", 'wb') as out:
        pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)
    
    with open(model_name+"/best_indexes.pkl", 'wb') as out:
        pickle.dump(best_indexes, out, pickle.HIGHEST_PROTOCOL)
            
    with open(model_name+"/fitted_models.pkl", 'wb') as out:
        pickle.dump(fitted_models, out, pickle.HIGHEST_PROTOCOL)
        
def load_results(model_name:str):
    
    with open(model_name+"/results.pkl", 'rb') as inp:
        results=pickle.load(inp)
    
    with open(model_name+"/best_indexes.pkl", 'rb') as inp:
        best_indexes=pickle.load(inp)
            
    with open(model_name+"/fitted_models.pkl", 'rb') as inp:
        fitted_models=pickle.load(inp)
        
    return results,best_indexes,fitted_models
    

def load_PCA_dfs(n:int)->Tuple[dict,pd.Series]:
    
    y = pd.read_parquet('dataset/y.parquet').squeeze()
    
    dfs={}
    
    for i in tqdm(range(2,n+1,10)):
        
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
                return_all_fitted:bool,
                hyperparameter_name:str,
                hyperparameter_values:List[float]) -> Tuple[dict,dict,Union[Type[ClusterMixin],Type[BaseEstimator]]]:
    
    results={}
    best_indexes={}
    fitted_estimators={}

    #dfs.keys()
    
    for dim in tqdm(dfs.keys(),desc="Total result"):
        
        get_estimator = (dim==2) or (return_all_fitted)
        
        results[dim],best_indexes[dim],fitted_estimator=hyperparameter_tuning("PCA_"+str(dim),
                                                                            estimator,hyperparameter_name,
                                                                            hyperparameter_values,
                                                                            dfs[dim], y,
                                                                            get_estimator)
        if (dim==2) or (return_all_fitted):
            fitted_estimators[dim]=copy.deepcopy(fitted_estimator)
        
    return results, best_indexes, fitted_estimators

def best_pca_dimension(results:dict,best_indexes:dict,hyperparameter_name:str):
    
    max_=0
    best_pca=None

    for x in range(2,200,10):
        s=results[x].iloc[best_indexes[x]]
        hyperparameter=s[hyperparameter_name]
        rand_index=s["rand index"]
        print(f"for PCA dimension {x}, the best {hyperparameter_name} hyperparameter is { hyperparameter } with a rand score of {rand_index}")
        
        if rand_index>max_:
            max_=rand_index
            best_pca=x

    print(f"\nThe best PCA dimension is {best_pca} with a rand score of {max_}")

def plot_clustering(X, labels, n_clusters_, cluster_centers=None):

    plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters_)]
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        
        plt.scatter(X[my_members]["PC_1"], X[my_members]["PC_2"], marker=markers[k%len(markers)], color=col)
        
        if cluster_centers is not None:
            cluster_center = cluster_centers[k]
            plt.scatter(cluster_center[0],cluster_center[1],marker=markers[k%len(markers)], edgecolor="black", s=200, color =col)
        
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    
def plot_images_per_cluster(type:str, dfs, fitted_estimators):
    
    for key in dfs.keys():
        
        labels=fitted_estimators[key].predict(dfs[key])
        
        if type=="GaussianMixture":
            n_clusters_=fitted_estimators[key].get_params()["n_components"]
        
        if type=="normalize_cut":
            n_clusters_=fitted_estimators[key].get_params()["n_clusters"]
        
        if type=="mean_shift":
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
    
        fig,axs=plt.subplots(n_clusters_,4 ,figsize=(4*3,n_clusters_*3))
    
        fig.suptitle(f"Digits plot for PCA dimension {key} with {n_clusters_} clusters")
    
        for item in [item for sublist in axs for item in sublist]:
            item.set_yticklabels([])
            item.set_xticklabels([])
            
        with open("pca/pca_"+str(key)+".pkl", 'rb') as inp:
            pca=pickle.load(inp)
            
        for k in range(n_clusters_):
            my_members = labels == k
            
            data=pca.inverse_transform(dfs[key][my_members])
        
            if data.shape[0]>=4:
                random_indexes=np.random.choice(data.shape[0], size=4, replace=False)
            else:
                #print("si")
                random_indexes=np.random.choice(data.shape[0], size=data.shape[0], replace=False)
                
            for i,imag in enumerate(data[random_indexes, :]):
                    axs[k,i].imshow(imag.reshape(28, 28))
                    
            if data.shape[0]==0:
                i=-1
        
            for j in range(i+1,4):
                axs[k,j].imshow(np.zeros(28*28).reshape(28,28))
                
        for ax,c in zip(axs[:,0],range(n_clusters_)):
            ax.set_ylabel(str(c), rotation=0, size='large')
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.show()
                
    