import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def set_w_intercept(w, use_CV, cache):
# def set_intercept(self, X_offset, y_offset, X_scale, w):
    """scale w, set intercept
    """
    if use_CV:
        i_fold = cache["i_fold"]
        X_offset = np.array(cache["mean_X_folds"][i_fold], dtype='float64')
        y_offset = cache["mean_y_folds"][i_fold]
        X_scale = np.array(cache["scale_X_folds"][i_fold], dtype='float64')        
    else:
        X_offset = np.array(cache["mean_X_train"], dtype='float64')
        y_offset = cache["mean_y_train"]
        X_scale = np.array(cache["scale_X_train"], dtype='float64')

    w = w / X_scale
    intercept = y_offset - np.dot(X_offset, w)
    
    return w.tolist(), intercept


def predict(X, w, intercept):
    """Predict using the linear model.
    """
    return np.matmul(X, w) + intercept


def squared_error(a, b):
    """a, b can be scalar or vector
    """
    return sum(np.square(a - b))


def agg_MSE(input, n_samples):
    return sum(site_dict["se_local"] for site, site_dict in input.items()) / n_samples

def agg_R2(input=input):
    numerator = sum(site_dict["se_local"] for site, site_dict in input.items())
    denominator = sum(site_dict["se_denominator_local"] for site, site_dict in input.items())
    if denominator == 0:
        raise Exception("agg_R2: demonimator is 0")
    return 1- numerator / denominator


def set_rank(w, name_features):
    w = np.array(w)
    name_features = np.array(name_features)

    index = np.argsort(np.abs(w))[::-1]

    if w.ndim == 1:
        w = w[:, np.newaxis]
    if name_features.ndim == 1:
        name_features = name_features[:, np.newaxis]       

    return np.concatenate((w[index], name_features[index]), axis=1).tolist()
