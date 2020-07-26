import os
import ujson as json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def idle(args, msg):
    output_dict = {"msg": msg}
    result_dict = {"output": output_dict}
    return json.dumps(result_dict)    

def readin(args):
    input = args["input"]
    state = args["state"]
    cache = args["cache"]
    cache_dir = state["cacheDirectory"]
    id = state["clientId"]
    owner = state["owner"] if "owner" in state else "local0" 
    return input, state, cache, cache_dir, id, owner


def split_save_train_test(input, output, cache, id, owner, cache_dir, X, y):
    # split to train / test dataset
    train_split_owner = input.get("train_split_owner", 0.8)
    train_split_local = input.get("train_split_local", 0.8)
    if train_split_owner >= 1 and train_split_local >= 1:  # all train no test data 
        raise ValueError("No test data, adjust train_split_owner/local")
    elif train_split_owner <= 0 and train_split_local <= 0:  # all test no train data 
        raise ValueError("No train data, adjust train_split_owner/local")        

    if id == owner:
        train_split = train_split_owner
    else:
        train_split = train_split_local

    if train_split >= 1:  # all train data
        train_index = np.arange(X.shape[0])
        test_index = np.array([])
        X_train = X
        X_test = np.array([])
        y_train = y
        y_test = np.array([])        
    elif train_split <= 0:  # all test data
        train_index = np.arange([])
        test_index = np.arange(X.shape[0])
        X_train = np.array([])
        X_test = X
        y_train = np.array([])
        y_test = y         
    else: #  split train / test data
        train_index, test_index = train_test_split(
            np.arange(X.shape[0]),  
            train_size=train_split, 
            random_state=input["random_state"], 
            shuffle=input["shuffle"]
        )
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

    output["train_index_local"] = train_index.tolist()
    output["test_index_local"] = test_index.tolist()

    cache["train_split"] = train_split
    
    np.save(os.path.join(cache_dir, "X_train.npy"), X_train)
    np.save(os.path.join(cache_dir, "X_test.npy"), X_test)
    np.save(os.path.join(cache_dir, "y_train.npy"), y_train)
    np.save(os.path.join(cache_dir, "y_test.npy"), y_test)

    return X_train, X_test, y_train, y_test


def split_folds_save_valid(input, output, cache, cache_dir, X_train, y_train):
    fold_indices = []
    valid_indices = []

    if cache["train_split"] > 0:  # has train data
        kf = KFold(
            n_splits=input["n_folds"], 
            shuffle=input["shuffle_CV"],
            random_state=input["random_state_CV"]    
        )

        for ii, (fold_index, valid_index) in enumerate(kf.split(X_train)):
            fold_indices.append(fold_index.tolist())
            valid_indices.append(valid_index.tolist())
            np.save(os.path.join(cache_dir, "X_valid_" + str(ii) + ".npy"), X_train[valid_index])        
            np.save(os.path.join(cache_dir, "y_valid_" + str(ii) + ".npy"), y_train[valid_index])

    output["fold_indices_local"] = fold_indices
    output["valid_indices_local"] = valid_indices
    cache["fold_indices"] = fold_indices
    # raise Exception("\n" + str(fold_indices) + "\n" +str(valid_indices))
    return fold_indices, valid_indices

class Stats():
    def __init__(self, preprocess_method):
        self.n_samples_train = 0
        self.n_samples_test = 0        
        self.sum_X_train = []  # not for "none"
        # self.sqsum_X_train = []
        self.sum_y_train = 0
        self.sum_y_test = 0
        self.preprocess_method = preprocess_method

    def cal_stats(self, X_train, y_train, y_test):
        self.n_samples_train = X_train.shape[0]
        self.n_samples_test = y_test.shape[0]
        # self.sqsum_X_train = np.sum(np.square(X_train), axis=0).tolist()             
        self.sum_y_train = sum(y_train) 
        self.sum_y_test = sum(y_test)  # for r2 score 

        if self.preprocess_method != "none":        
            self.sum_X_train = np.sum(X_train, axis=0).tolist()
   
    def add_output(self, dict):
        dict["n_samples_train_local"] = self.n_samples_train
        dict["n_samples_test_local"] = self.n_samples_test        
        # dict["sqsum_X_train_local"] = self.sqsum_X_train
        dict["sum_y_train_local"] = self.sum_y_train
        dict["sum_y_test_local"] = self.sum_y_test  

        if self.preprocess_method != "none":  
            dict["sum_X_train_local"] = self.sum_X_train
          

class Stats_CV():
    def __init__(self, preprocess_method):
        self.n_samples_folds = []
        self.n_samples_valids = []
        self.sum_X_folds = []  # list of list
        # self.sqsum_X_folds = []  # list of list
        self.sum_y_folds = []
        self.sum_y_valids = []
        self.preprocess_method = preprocess_method

    def cal_stats(self, X_train, y_train, fold_indices, valid_indices):
        for fold_index, valid_index in zip(fold_indices, valid_indices):
            self.n_samples_folds.append(len(fold_index))
            self.n_samples_valids.append(len(valid_index))
            # self.sqsum_X_folds.append(np.sum(np.square(X_train[fold_index]), axis=0).tolist())
            self.sum_y_folds.append(sum(y_train[fold_index]))
            self.sum_y_valids.append(sum(y_train[valid_index]))  

            if self.preprocess_method != "none": 
                self.sum_X_folds.append(np.sum(X_train[fold_index], axis=0).tolist())
           
    def add_output(self, dict):
        dict["n_samples_folds_local"] = self.n_samples_folds
        dict["n_samples_valids_local"] = self.n_samples_valids
        # dict["sqsum_X_folds_local"] = self.sqsum_X_folds            
        dict["sum_y_folds_local"] = self.sum_y_folds
        dict["sum_y_valids_local"] = self.sum_y_valids  

        if self.preprocess_method != "none":
            dict["sum_X_folds_local"] = self.sum_X_folds
        # raise Exception(self.sum_X_folds)
              
def preprocess_save_calXy(input, cache_dir, X_train, y_train):
    X_train = X_train.copy()
    y_train = y_train.copy()

    # center X, y
    X_train = X_train - input["mean_X_train"]
    y_train = y_train - input["mean_y_train"]
    X_train = X_train / input["scale_X_train"]
    # raise Exception(X_train)
    np.save(os.path.join(cache_dir, "X_train.npy"), X_train)
    np.save(os.path.join(cache_dir, "y_train.npy"), y_train)

    Xy = np.dot(X_train.T, y_train).tolist()
    return Xy


def preprocess_save_CV(input, cache_dir, X_train, y_train, fold_indices):
    # Xy_folds = []
    for ii, train_index in enumerate(fold_indices):
        X_fold = X_train[train_index]
        y_fold = y_train[train_index]

        # center X, y
        X_fold = X_fold - input["mean_X_folds"][ii]
        y_fold = y_fold - input["mean_y_folds"][ii]
        X_fold = X_fold / input["scale_X_folds"][ii]
        
        np.save(os.path.join(cache_dir, "X_fold_" + str(ii) + ".npy"), X_fold)        
        np.save(os.path.join(cache_dir, "y_fold_" + str(ii) + ".npy"), y_fold)
    #     Xy = np.dot(X_fold.T, y_fold).tolist()
    #     Xy_folds.append(Xy)
    
    # return Xy_folds








# def scaler(X, y, method):
#     X = X.copy()
#     y = np.asarray(y, dtype=X.dtype)
#     n_features = X.shape[1]

#     if method == "none":
#         X_offset = np.zeros(n_features, dtype=X.dtype)
#         y_offset = np.zeros(n_features, dtype=X.dtype)
#         X_scale = np.ones(n_features, dtype=X.dtype)
#     elif method == "normalize":
#         X_offset = np.mean(X, axis=0, dtype=X.dtype)
#         X -= X_offset

#             X, X_scale = f_normalize(X, axis=0, copy=False,
#                                         return_norm=True)



#     return X, y, X_offset, y_offset, X_scale

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, 
    #     train_size=train_split, 
    #     random_state=input["random_state"], 
    #     shuffle=input["shuffle"]
    # ) 
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, 
    #     test_size=10, 
    #     random_state=input["random_state"], 
    #     shuffle=input["shuffle"]
    # )     
