import os

import numpy as np
import ujson as json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def readin_remote(args):
    """Extracts frequently-used dicts and variables from COINSTAC pipeline dict.

    Args:
        args: COINSTAC pipeline dict.

    Returns:
        input, state, cache: COINSTAC dict.
    """
    input = args["input"]
    state = args["state"]
    cache = args["cache"]
    return input, state, cache


def receive_params(input_owner, cache):
    """Receives parameters from owner site and puts them in cache dict.

    Args:
        input_owner: input dict of owner site.
        cache: COINSTAC dict.
    """
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


class Stats_agg:
    """Stats of the whole, an aggregation of the stats from local sites.

    Attributes:
        n_samples_train: int, the number of samples in the training dataset.
        n_samples_test: int.
        mean_X_train: float list, len = n_features, column-wise mean of X_train.
        mean_y_train: float, mean of y_train.
        mean_y_test: float, mean of y_test.
        preprocessed_method: str, preprocessing method applied on the training 
            dataset.
        n_features: int, the number of features.
        train_index: dict, {local site (str): index of X for X_train (list)}, 
            for reconstruction of the train/test datasets after the program 
            ends.  
        test_index: dict, {local site (str): index of X for X_test (list)}.  
    """

    def __init__(self, preprocess_method, n_features):
        """Initiates Stats_agg.

        Args:
            preprocess_method: str.
            n_features: int, the number of features.
        """
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
        """Aggregates stats from local sites.

        Args:
            input: COINSTAC dict.
        """
        # agg n_samples_train/test
        n_samples_train = sum(
            site_dict["n_samples_train_local"]
            for site, site_dict in input.items()
        )
        self.n_samples_train = int(n_samples_train)
        n_samples_test = sum(
            site_dict["n_samples_test_local"]
            for site, site_dict in input.items()
        )
        self.n_samples_test = int(n_samples_test)

        # agg mean_y_train/test
        if self.preprocess_method != "none":
            self.mean_y_train = (
                sum(
                    site_dict["sum_y_train_local"]
                    for site, site_dict in input.items()
                )
                / n_samples_train
            )
        else:
            self.mean_y_train = 0.0

        self.mean_y_test = (
            sum(
                site_dict["sum_y_test_local"]
                for site, site_dict in input.items()
            )
            / n_samples_test
        )

        # set mean_X_train
        if self.preprocess_method != "none":
            sum_X_train = np.zeros(self.n_features)
            for site, site_dict in input.items():
                sum_X_train = sum_X_train + site_dict["sum_X_train_local"]
            self.mean_X_train = (sum_X_train / n_samples_train).tolist()
        else:
            self.mean_X_train = (np.zeros(self.n_features)).tolist()

        # train /test index
        for site, site_dict in input.items():
            self.train_index[str(site)] = site_dict["train_index_local"]
            self.test_index[str(site)] = site_dict["test_index_local"]

    def add_output(self, output):
        """Adds attributes and message to the output dict.

        Args:
            output: COINSTAC output dict.
        """
        output["msg"] = "to_sqsum"
        output["mean_X_train"] = self.mean_X_train

    def add_cache(self, cache):
        """Adds attributes to the cache dict.

        Args:
            cache: COINSTAC cache dict.
        """
        cache["n_samples_train"] = self.n_samples_train
        cache["n_samples_test"] = self.n_samples_test
        cache["mean_X_train"] = self.mean_X_train
        cache["mean_y_train"] = self.mean_y_train
        cache["mean_y_test"] = self.mean_y_test
        cache["train_index"] = self.train_index
        cache["test_index"] = self.test_index


class Stats_agg_CV:
    """Stats of the whole for CV, an aggregation of the stats from local sites.

    Attributes:
        n_samples_folds: int list, len = n_folds, the number of samples in the 
            training dataset of each fold.
        n_samples_valids: int list, len = n_folds, the number of samples in the 
            validation dataset of each fold.
        mean_X_folds: float list of 2D-list, shape (n_folds, (n_features,)), 
            column-wise mean of X_fold.
        mean_y_folds: float list, len = n_folds, mean of y_fold.
        mean_y_valids: float list, len = n_folds, sum of y_valid.
        preprocessed_method: str, preprocessing method applied on the training 
            dataset.
        n_features: int, the number of features.
        n_folds: int, the number of folds in cross validation.
        fold_indices: dict, 
            {local site (str): index of X_train for X_fold (list)}, 
            for reconstruction of the fold/valid datasets after the program 
            ends.  
        valid_indices: dict, 
            {local site (str): index of X_train for X_valid (list)}.  
    """

    def __init__(self, preprocess_method, n_features, n_folds):
        """Initiates Stats_agg_CV.

        Args: 
            preprocess_method: str.
            n_features: int, the number of features.
            n_folds: int, the number of folds for CV.
        """
        self.n_samples_folds = []
        self.n_samples_valids = []
        self.mean_X_folds = []  # a list of lists
        self.mean_y_folds = []
        self.mean_y_valids = []
        self.preprocess_method = preprocess_method
        self.n_features = n_features
        self.n_folds = n_folds
        self.fold_indices = {}
        self.valid_indices = {}

    def agg_stats(self, input):
        """Aggregates stats from local sites.

        Args:
            input: COINSTAC dict.
        """
        # agg n_samples_folds, n_samples_valids
        for ii in range(self.n_folds):
            n_samples = sum(
                site_dict["n_samples_folds_local"][ii]
                for site, site_dict in input.items()
            )
            self.n_samples_folds.append(int(n_samples))
        for ii in range(self.n_folds):
            n_samples = sum(
                site_dict["n_samples_valids_local"][ii]
                for site, site_dict in input.items()
            )
            self.n_samples_valids.append(int(n_samples))

        # agg mean_y_folds, mean_y_valids
        if self.preprocess_method != "none":
            for ii in range(self.n_folds):
                mean_y = (
                    sum(
                        site_dict["sum_y_folds_local"][ii]
                        for site, site_dict in input.items()
                    )
                    / self.n_samples_folds[ii]
                )
                self.mean_y_folds.append(mean_y)
        else:
            self.mean_y_folds = np.zeros(self.n_folds).tolist()

        for ii in range(self.n_folds):
            mean_y_valid = (
                sum(
                    site_dict["sum_y_valids_local"][ii]
                    for site, site_dict in input.items()
                )
                / self.n_samples_valids[ii]
            )
            self.mean_y_valids.append(mean_y_valid)

        # set mean_X_folds
        if self.preprocess_method != "none":
            for ii in range(self.n_folds):
                sum_X = np.zeros(self.n_features)
                for site, site_dict in input.items():
                    sum_X = sum_X + site_dict["sum_X_folds_local"][ii]
                mean_X = sum_X / self.n_samples_folds[ii]
                self.mean_X_folds.append(mean_X.tolist())
        else:
            self.mean_X_folds = np.tile(
                np.zeros(self.n_features), (self.n_folds, 1)
            ).tolist()

        # fold / valid indices
        for site, site_dict in input.items():
            self.fold_indices[str(site)] = site_dict["fold_indices_local"]
            self.valid_indices[str(site)] = site_dict["valid_indices_local"]

    def add_output(self, output):
        """Adds attributes and message to the output dict.

        Args:
            output: COINSTAC output dict.
        """
        output["msg"] = "to_sqsum"
        output["mean_X_folds"] = self.mean_X_folds

    def add_cache(self, cache):
        """Adds attributes to the cache dict.

        Args:
            cache: COINSTAC cache dict.
        """
        cache["n_samples_folds"] = self.n_samples_folds
        cache["n_samples_valids"] = self.n_samples_valids
        cache["mean_X_folds"] = self.mean_X_folds
        cache["mean_y_folds"] = self.mean_y_folds
        cache["mean_y_valids"] = self.mean_y_valids
        cache["fold_indices"] = self.fold_indices
        cache["valid_indices"] = self.valid_indices


def agg_scale_sqsum(input, preprocess_method, n_features, n_samples):
    """Aggregates from locals for sum of squares and scale of the whole X_train.

    Sum of sqaures of the whole will be used in coordinate descent.
    Scale of the whole will be used in preprocessing and setting w and 
        intercept.

    Args:
        input: COINSTAC dict.
        preprocessed_method: str, preprocessing method applied on the training 
            dataset.        
        n_features: int, the number of features.  
        n_samples: int, the number of samples in X_train. 

    Returns:
        scale_X_train: list, len = n_features, column-wise scale of X_train.
        sqsum_X_train: list, len = n_features, column-wise sum of squares of 
            X_train.
    """
    sqsum_X_train = np.zeros(n_features)
    for site, site_dict in input.items():
        sqsum_X_train = sqsum_X_train + site_dict["sqsum_X_train_local"]

    # agg scale_X_train, for preprocess and intercept
    if preprocess_method == "standardize":
        scale_X_train = np.sqrt(sqsum_X_train / (n_samples - 1))
    elif preprocess_method == "normalize":
        scale_X_train = np.sqrt(sqsum_X_train)
    elif preprocess_method in ["center", "none"]:
        scale_X_train = np.ones(n_features)

    # agg sqsum_X_train, for coordinate descent. no need for ["center", "none"]
    if preprocess_method == "standardize":
        scale_nonzero = np.where(scale_X_train == 0.0, 1.0, scale_X_train)
        sqsum_X_train = sqsum_X_train / (scale_nonzero * scale_nonzero)
    elif preprocess_method == "normalize":
        sqsum_X_train = np.ones(n_features)

    return scale_X_train.tolist(), sqsum_X_train.tolist()


def agg_scale_sqsum_CV(
    input, preprocess_method, n_features, n_samples_folds, n_folds
):
    """Aggregates from locals for sum of squares and scale of whole X_fold.

    Sum of sqaures of the whole will be used in coordinate descent.
    Scale of the whole will be used in preprocessing and setting w and 
        intercept.

    Args:
        input: COINSTAC dict.
        preprocessed_method: str, preprocessing method applied on the training 
            dataset.        
        n_features: int, the number of features.  
        n_samples_folds: int list, len = n_folds, the number of samples in 
            X_fold. 
        n_folds: int, the number of folds in cross validation.

    Returns:
        scale_X_folds: a list of lists, shape (n_folds, (n_features,)), 
            column-wise scale of X_fold.
        sqsum_X_folds: a list of lists, shape (n_folds, (n_features,)), 
            column-wise sum of squares of X_fold.
    """
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
            scale_X_folds.append(
                np.sqrt(
                    np.array(sqsum_X_folds[ii]) / (n_samples_folds[ii] - 1)
                ).tolist()
            )
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
            scale = np.where(scale == 0.0, 1.0, scale)
            sqsum = sqsum / (scale * scale)
            sqsum_X_folds.append(sqsum.tolist())
    elif preprocess_method == "normalize":
        sqsum_X_folds = np.tile(np.ones(n_features), (n_folds, 1)).tolist()

    return scale_X_folds, sqsum_X_folds


def set_lambdas(lambdas, input, cache):
    """Sets lambdas, a list of L1-regularizer.

    If the user-input lambdas is empty, generates one in decreasing order. 
        Else, sorts it in decreasing order.

    Args:
        lambdas: float list, input by user.
        input, cache: COINSTAC dict. 

    Returns:
        lambdas: float list.    
    """
    if not lambdas:  # lambdas = []
        # agg Xy
        n_features = cache["n_features"]
        Xy = np.zeros(n_features)
        for site, site_dict in input.items():
            Xy = Xy + site_dict["Xy"]
        # generate lambdas
        lambdas = lambda_grid(
            Xy=Xy,
            eps=cache["eps"],
            n_lambdas=cache["n_lambdas"],
            n_samples=cache["n_samples_train"],
        )
    else:
        # make sure lambdas is properly ordered.
        lambdas = np.sort(lambdas)[::-1]

    return lambdas.tolist()


def lambda_grid(Xy, eps, n_lambdas, n_samples):
    """Generates a grid of lambda values. 
    
    In the grid, the max is lambda_max (i.e., if lambda >= lambda_max, w = 0) 
        which will be automatically calculated.

    Args:
        Xy: np.ndarray of shape (n_features,).
        eps: float, lambda_min = eps * lambda_max.
        n_lambdas: int, the number of lambda.
        n_samples: int, the number of samples in the training dataset.

    Returns:
        np.ndarray of shape (n_lambdas,) on a log sacle in decreasing order. 
    """
    lambda_max = abs(Xy).max() / n_samples

    if lambda_max <= np.finfo(float).resolution:
        lambdas = np.array([np.finfo(float).resolution])
        return lambdas

    return np.logspace(
        np.log10(lambda_max * eps), np.log10(lambda_max), num=n_lambdas
    )[::-1]


def pick_jth_feature(selection, n_features, i_feature):
    """Pick next feature's coefficient to be updated in coordinate descent.

    Args:
        selection: str, "random" or "cyclic", the sequence of updating 
            coefficients in an iteration.
        n_feature: int, the number of features.
        i_feature: int, if selection == "cyclic", then next feature is the 
            i_feature-th feature.

    Returns:
        jj: int, the jj-th feature.
    """
    if selection == "random":
        rng = np.random.default_rng()
        jj = int(rng.integers(n_features))  # np.int64 to int
    else:
        jj = i_feature

    return jj
