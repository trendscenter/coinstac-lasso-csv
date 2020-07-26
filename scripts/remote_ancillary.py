import os
import ujson as json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def readin_remote(args):
    input = args["input"]
    state = args["state"]
    cache = args["cache"]
    return input, state, cache

def receive_params(input_owner, cache):
    cache["label"] = input_owner["label"]    
    cache["train_split_local"] = input_owner["train_split_local"]
    cache["train_split_owner"] = input_owner["train_split_owner"]    
    cache["preprocess_method"] = input_owner["preprocess_method"]
    cache["max_iter"] = input_owner["max_iter"]
    cache["tol"] = input_owner["tol"]
    cache["positive"] = input_owner["positive"]
    cache["selection"] = input_owner["selection"]
    cache["lambdas"] = input_owner["lambdas"]
    cache["eps"] = input_owner["eps"]
    cache["n_lambdas"] = input_owner["n_lambdas"]
    cache["use_CV"] = input_owner["use_CV"]

    if cache["use_CV"]:
        cache["n_folds"] = input_owner["n_folds"]    

    cache["name_features"] = input_owner["name_features"]
    cache["n_features"] = len(cache["name_features"])    


class Stats_agg():
    def __init__(self, preprocess_method, n_features):
        self.n_samples_train = 0
        self.n_samples_test = 0        
        self.mean_X_train = []
        self.mean_y_train = 0
        self.mean_y_test = 0
        self.preprocess_method = preprocess_method    
        self.n_features = n_features
        self.train_index = {}
        self.test_index = {}

    def agg_stats(self, input):
        # agg n_samples_train/test
        n_samples_train = sum(site_dict["n_samples_train_local"] for site, site_dict in input.items())
        self.n_samples_train = int(n_samples_train)
        n_samples_test = sum(site_dict["n_samples_test_local"] for site, site_dict in input.items())
        self.n_samples_test = int(n_samples_test)        
                  
        # agg mean_y_train/test
        if self.preprocess_method != "none":
            self.mean_y_train = sum(site_dict["sum_y_train_local"] for site, site_dict in input.items()) / n_samples_train
        else:
            self.mean_y_train = 0.0

        self.mean_y_test = sum(site_dict["sum_y_test_local"] for site, site_dict in input.items()) / n_samples_test   
        
        # set mean_X_train
        if self.preprocess_method != "none":
            sum_X_train = np.zeros(self.n_features)
            for site, site_dict in input.items():
                sum_X_train = sum_X_train + site_dict.get("sum_X_train_local", np.zeros(self.n_features))
            self.mean_X_train = (sum_X_train / n_samples_train).tolist()
        else:
            self.mean_X_train = (np.zeros(self.n_features)).tolist() 

        # train /test index
        for site, site_dict in input.items():
            self.train_index[str(site)] = site_dict["train_index_local"]
            self.test_index[str(site)] = site_dict["test_index_local"]

    def add_output(self, output):    
            output["msg"] = "to_sqsum"
            output["mean_X_train"] = self.mean_X_train         

    def add_cache(self, cache):
        cache["n_samples_train"] = self.n_samples_train    
        cache["n_samples_test"] = self.n_samples_test        
        cache["mean_X_train"] = self.mean_X_train     
        cache["mean_y_train"] = self.mean_y_train
        cache["mean_y_test"] = self.mean_y_test
        cache["train_index"] = self.train_index
        cache["test_index"] = self.test_index


class Stats_agg_CV():
    def __init__(self, preprocess_method, n_features, n_folds):
        self.n_samples_folds = []  
        self.n_samples_valids = []
        self.mean_X_folds = []  # list of list
        self.mean_y_folds = []
        self.mean_y_valids = []
        self.preprocess_method = preprocess_method    
        self.n_features = n_features
        self.n_folds = n_folds
        self.fold_indices = {}
        self.valid_indices = {}        

    def agg_stats(self, input):
        # agg n_samples_folds, n_samples_valids
        for ii in range(self.n_folds):
            n_samples = sum(site_dict["n_samples_folds_local"][ii] for site, site_dict in input.items())
            self.n_samples_folds.append(int(n_samples))    
        for ii in range(self.n_folds):            
            n_samples = sum(site_dict["n_samples_valids_local"][ii] for site, site_dict in input.items())
            self.n_samples_valids.append(int(n_samples))    

        # agg mean_y_folds, mean_y_valids
        if self.preprocess_method != "none":
            for ii in range(self.n_folds): 
                mean_y = sum(site_dict["sum_y_folds_local"][ii] for site, site_dict in input.items()) / self.n_samples_folds[ii]
                self.mean_y_folds.append(mean_y)
        else:
            self.mean_y_folds = np.zeros(self.n_folds).tolist()                               

        for ii in range(self.n_folds):             
            mean_y_valid = sum(site_dict["sum_y_valids_local"][ii] for site, site_dict in input.items()) / self.n_samples_valids[ii]
            self.mean_y_valids.append(mean_y_valid)  
        
        # set mean_X_folds
        if self.preprocess_method != "none":
            for ii in range(self.n_folds):    
                sum_X = np.zeros(self.n_features)
                for site, site_dict in input.items():
                    # raise Exception(site_dict["sum_X_folds_local"])
                    sum_X = sum_X + site_dict["sum_X_folds_local"][ii]
                mean_X = sum_X / self.n_samples_folds[ii]
                self.mean_X_folds.append(mean_X.tolist())
        else:
            self.mean_X_folds = np.tile(np.zeros(self.n_features), (self.n_folds, 1)).tolist()   

        # fold / valid indices
        for site, site_dict in input.items():
            self.fold_indices[str(site)] = site_dict["fold_indices_local"]
            self.valid_indices[str(site)] = site_dict["valid_indices_local"]        

    def add_output(self, output):    
        output["msg"] = "to_sqsum"
        output["mean_X_folds"] = self.mean_X_folds         
        # output["mean_y_folds"] = self.mean_y_folds
        
    def add_cache(self, cache):
        cache["n_samples_folds"] = self.n_samples_folds    
        cache["n_samples_valids"] = self.n_samples_valids   
        # raise Exception("\n" + str(cache["n_samples_valids"])+"\n"+str(cache["n_samples_folds"]))      
        cache["mean_X_folds"] = self.mean_X_folds 
        cache["mean_y_folds"] = self.mean_y_folds
        cache["mean_y_valids"] = self.mean_y_valids
        # raise Exception(cache["mean_y_valids"])
        cache["fold_indices"] = self.fold_indices
        cache["valid_indices"] = self.valid_indices



def agg_scale_sqsum(input, preprocess_method, n_features, n_samples):
    sqsum_X_train = np.zeros(n_features)
    for site, site_dict in input.items():
        sqsum_X_train = sqsum_X_train + site_dict.get("sqsum_X_train_local", np.zeros(n_features))    

    # agg scale_X_train, for preprocess and intercept     
    if preprocess_method == "standardize":
        scale_X_train = np.sqrt(sqsum_X_train / (n_samples - 1))
    elif preprocess_method == "normalize":
        scale_X_train = np.sqrt(sqsum_X_train)
    elif preprocess_method in ["center", "none"]:
        scale_X_train = np.ones(n_features)

    # agg sqsum_X_train, for coordinate descent. no need for ["center", "none"] 
    if preprocess_method == "standardize":
        sqsum_X_train = sqsum_X_train / (scale_X_train * scale_X_train)
    elif preprocess_method == "normalize":
        sqsum_X_train = np.ones(n_features)

    return scale_X_train.tolist(), sqsum_X_train.tolist()

def agg_scale_sqsum_CV(input, preprocess_method, n_features, n_samples_folds, n_folds):
    sqsum_X_folds = []   
    for ii in range(n_folds):    
        sqsum_X = np.zeros(n_features)
        for site, site_dict in input.items():
            sqsum_X = sqsum_X + site_dict["sqsum_X_folds_local"][ii]
        sqsum_X_folds.append(sqsum_X.tolist())

    # agg scale_X_folds, for preprocess and intercept     
    if preprocess_method == "standardize":
        scale_X_folds = []
        for ii in range(n_folds):
            scale_X_folds.append(np.sqrt(np.array(sqsum_X_folds[ii]) / (n_samples_folds[ii] - 1)).tolist())
    elif preprocess_method == "normalize":
        scale_X_folds = []
        for ii in range(n_folds):
            scale_X_folds.append(np.sqrt(sqsum_X_folds[ii]).tolist())         
    elif preprocess_method in ["center", "none"]:
        scale_X_folds = np.tile(np.ones(n_features), (n_folds, 1)).tolist()

    # agg sqsum_X_folds, for coordinate descent. no need for ["center", "none"]
    if preprocess_method == "standardize":
        sqsum_X_folds_old = sqsum_X_folds
        sqsum_X_folds = []
        for ii in range(n_folds):
            sqsum = np.array(sqsum_X_folds_old[ii]) 
            scale = np.array(scale_X_folds[ii])
            sqsum = sqsum / (scale * scale)
            sqsum_X_folds.append(sqsum.tolist()) 
    elif preprocess_method == "normalize":
        sqsum_X_folds = np.tile(np.ones(n_features), (n_folds, 1)).tolist()

    return scale_X_folds, sqsum_X_folds                        
               
    
def set_lambdas(lambdas, input, cache):
    if not lambdas:  # lambdas = []
        # agg Xy
        n_features = cache["n_features"]
        Xy = np.zeros(n_features)
        for site, site_dict in input.items():
            Xy = Xy + site_dict.get("Xy", np.zeros(n_features))   
        # generate lambdas       
        lambdas = lambda_grid(
            Xy=Xy,
            eps=cache['eps'], 
            n_lambdas=cache['n_lambdas'],
            n_samples=cache['n_samples_train']
        )
    else:
        # make sure lambdas is properly ordered.
        lambdas = np.sort(lambdas)[::-1]

    return lambdas.tolist()
    

def lambda_grid(Xy, eps, n_lambdas, n_samples):
    """ Xy : array-like of shape (n_features,) 
    Xy = np.dot(X.T, y) that can be precomputed.
    """
    lambda_max = abs(Xy).max() / n_samples

    if lambda_max <= np.finfo(float).resolution:
        lambdas = np.array([np.finfo(float).resolution])
        return lambdas

    return np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max),
                       num=n_lambdas)[::-1]
                    
def pick_jth_feature(selection, n_features, i_feature):        
    if selection == "random":
        rng = np.random.default_rng()
        jj = int(rng.integers(n_features))  # np.int64 to int
    else:
        jj = i_feature

    return jj



      