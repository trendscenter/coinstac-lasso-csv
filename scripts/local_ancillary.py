import os

import numpy as np
import ujson as json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
 

def readin(args):
    """Extracts frequently-used dicts and variables from COINSTAC pipeline dict.

    Args:
        args: COINSTAC pipeline dict.

    Returns:
        input, state, cache: COINSTAC dict.
        cache_dir: str, cache directory.
        id: str, the local site's id.
        owner: str, the owner site's id.

    Raises:
        ValueError: if train_split_owner >= 1 and train_split_local >= 1,  all train no test data for the whole.
        ValueError: if train_split_owner <= 0 and train_split_local <= 0, all test no train data for the whole.
    """
    input = args["input"]
    state = args["state"]
    cache = args["cache"]
    cache_dir = state["cacheDirectory"]
    id = state["clientId"]
    owner = state["owner"] if "owner" in state else "local0" 
    return input, state, cache, cache_dir, id, owner


def split_save_train_test(input, output, cache, id, owner, cache_dir, X, y):
    """ Splits raw data to train/test datasets and save the datasets.

    Args:
        input, output, cache: COINSTAC dict.
        cache_dir: cache directory.
        id: str, the local site's id.
        owner: str, the owner site's id.       
        X: np.ndarray of shape (n_samples, n_features).
        y: np.ndarray of shape (n_features,).

    Returns:
        X_train, X_test: np.ndarray of shape (n_samples, n_features).
        y_train, y_test: np.ndarray of shape (n_features,).
    """
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
        train_index = np.array([])
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
    """Splits train data to fold/valid datasets and save the valid datasets.

    Args:
        input, output, cache: COINSTAC dict.
        cache_dir: str, cache directory.      
        X_train: np.ndarray of shape (n_samples, n_features).
        y_train: np.ndarray of shape (n_features,).

    Returns:
        fold_indices: a list of lists, len = n_folds, indices of X/y_train for the training dataset in each fold.
        valid_indices: a list of lists, len = n_folds, indices of X/y_train for the validation dataset in each fold. 
    """    
    if cache["train_split"] > 0:  # has train data
        fold_indices = []
        valid_indices = []

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
    else:
        n_folds = input["n_folds"]
        fold_indices = np.tile(np.array([]), (n_folds, 1)).tolist()
        valid_indices = np.tile(np.array([]), (n_folds, 1)).tolist()            

    output["fold_indices_local"] = fold_indices
    output["valid_indices_local"] = valid_indices
    cache["fold_indices"] = fold_indices
    return fold_indices, valid_indices


class Stats():
    """Summation stats of train/test datasets at the local site.

    Attributes:
        n_samples_train: int, the number of samples in the training dataset.
        n_samples_test: int.
        sum_X_train: float np.ndarray of shape (n_features,), column-wise sum of X_train.
        sum_y_train: float, sum of y_train.
        sum_y_test: float, sum of y_test.
        preprocessed_method: str, preprocessing method applied on the training dataset.
    """

    def __init__(self, cache, preprocess_method):
        """Initiates Stats with cache (dict) and preprocess_method (str).
        """
        n_features = cache["n_features"]
        self.n_samples_train = 0
        self.n_samples_test = 0        
        self.sum_X_train = np.zeros(n_features).tolist()  # not for "none"
        self.sum_y_train = 0.0
        self.sum_y_test = 0.0
        self.preprocess_method = preprocess_method

    def cal_stats(self, X_train, y_train, y_test):
        """Calculates summation stats at the local site.

        Args:
            X_train: np.ndarray of shape (n_samples, n_features).
            y_train, y_test: np.ndarray of shape (n_features,).
        """        
        self.n_samples_train = X_train.shape[0]
        self.n_samples_test = y_test.shape[0]         
        self.sum_y_train = sum(y_train) 
        self.sum_y_test = sum(y_test)  # for r2 score 

        if self.preprocess_method != "none":        
            self.sum_X_train = np.sum(X_train, axis=0).tolist()
   
    def add_output(self, dict):
        """Adds attributes to the output dict.

        Args:
            dict: the output dict.
        """        
        dict["n_samples_train_local"] = self.n_samples_train
        dict["n_samples_test_local"] = self.n_samples_test        
        dict["sum_y_train_local"] = self.sum_y_train
        dict["sum_y_test_local"] = self.sum_y_test  

        if self.preprocess_method != "none":  
            dict["sum_X_train_local"] = self.sum_X_train
          

class Stats_CV():
    """Summation stats of fold/valid datasets for cross validation at the local site.

    Attributes:
        n_samples_folds: int list, len = n_folds, the number of samples in the training dataset in each fold.
        n_samples_valids: int list, len = n_folds, the number of samples in the validation dataset in each fold.
        sum_X_folds: float list of 2D-list, shape (n_folds, (n_features,)), column-wise sum of X_fold.
        sum_y_folds: float list, len = n_folds, sum of y_fold.
        sum_y_valids: float list, len = n_folds, sum of y_valid.
        preprocessed_method: str, preprocessing method applied on the training dataset.
    """    

    def __init__(self, cache, preprocess_method):
        """Initiates Stats_CV with cache (dict) and preprocess_method (str).
        """        
        n_folds = cache["n_folds"]
        n_features = cache["n_features"]
        self.n_samples_folds = np.zeros(n_folds).tolist()
        self.n_samples_valids = np.zeros(n_folds).tolist()
        self.sum_X_folds = np.tile(np.zeros(n_features), (n_folds, 1)).tolist()  # a list of lists, not for "none"
        self.sum_y_folds = np.zeros(n_folds).tolist()
        self.sum_y_valids = np.zeros(n_folds).tolist()
        self.preprocess_method = preprocess_method

    def cal_stats(self, X_train, y_train, fold_indices, valid_indices):
        """Calculates summation stats for folds at the local site.

        Args:
            X_train: np.ndarray of shape (n_samples, n_features).
            y_train: np.ndarray of shape (n_features,).   
            fold_indices: a list of lists, len = n_folds, indices of X/y_train for the training dataset in each fold.
            valid_indices: a list of lists, len = n_folds, indices of X/y_train for the validation dataset in each fold.         
        """         
        for ii, (fold_index, valid_index) in enumerate(zip(fold_indices, valid_indices)):
            self.n_samples_folds[ii] = len(fold_index)
            self.n_samples_valids[ii] = len(valid_index)
            self.sum_y_folds[ii] = sum(y_train[fold_index])
            self.sum_y_valids[ii] = sum(y_train[valid_index])

            if self.preprocess_method != "none": 
                self.sum_X_folds[ii] = np.sum(X_train[fold_index], axis=0).tolist()
    
    def add_output(self, dict):
        """Adds attributes to the output dict.

        Args:
            dict: the output dict.
        """             
        dict["n_samples_folds_local"] = self.n_samples_folds
        dict["n_samples_valids_local"] = self.n_samples_valids        
        dict["sum_y_folds_local"] = self.sum_y_folds
        dict["sum_y_valids_local"] = self.sum_y_valids  

        if self.preprocess_method != "none":
            dict["sum_X_folds_local"] = self.sum_X_folds

              
def preprocess_save_calXy(input, cache_dir, X_train, y_train):
    """Preprocesses the training dataset, save it and calculates Xy.

    Args:
        input: COINSTAC dict.
        cache_dir: str, cache directory.
        X_train: np.ndarray of shape (n_samples, n_features).
        y_train: np.ndarray of shape (n_features,).           

    Returns:
        Xy: float list, len = n_features. 
    """     
    X_train = X_train.copy()
    y_train = y_train.copy()

    # center X, y
    X_train = X_train - input["mean_X_train"]
    y_train = y_train - input["mean_y_train"]
    # scale X
    scale_X_train = np.array(input["scale_X_train"])
    scale_X_train = np.where(scale_X_train == 0.0, 1.0, scale_X_train)
    X_train = X_train / scale_X_train

    np.save(os.path.join(cache_dir, "X_train.npy"), X_train)
    np.save(os.path.join(cache_dir, "y_train.npy"), y_train)

    Xy = np.dot(X_train.T, y_train).tolist()
    return Xy


def preprocess_save_CV(input, cache_dir, X_train, y_train, fold_indices):
    """Preprocesses the fold datasets for cross validation and save them.

    Args:
        input: COINSTAC dict.
        cache_dir: str, cache directory.
        X_train: np.ndarray of shape (n_samples, n_features).
        y_train: np.ndarray of shape (n_features,).  
        fold_indices: a list of lists, len = n_folds, indices of X/y_train for the training dataset in each fold.                 
    """     
    for ii, train_index in enumerate(fold_indices):
        X_fold = X_train[train_index]
        y_fold = y_train[train_index]

        # center X, y
        X_fold = X_fold - input["mean_X_folds"][ii]
        y_fold = y_fold - input["mean_y_folds"][ii]
        # scale X
        scale_X_fold = np.array(input["scale_X_folds"][ii])
        scale_X_fold = np.where(scale_X_fold == 0.0, 1.0, scale_X_fold)        
        X_fold = X_fold / scale_X_fold
        
        np.save(os.path.join(cache_dir, "X_fold_" + str(ii) + ".npy"), X_fold)        
        np.save(os.path.join(cache_dir, "y_fold_" + str(ii) + ".npy"), y_fold)