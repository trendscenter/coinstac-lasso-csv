import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def set_w_intercept(w, use_CV, cache):
    """Scales slope w and calculates intercept for unpreprocessed raw data. 

    Args:
        w: float list, len = n_features, slope in the model y = wx + intercept,
            for preprocessed data. 
        use_CV: boolean, whether is using cross validation or not currently.
        cache: COINSTAC dict.

    Returns:
        w: float list, len = n_features, slope in the model y = wx + intercept, 
            for unpreprocessed raw data. 
        intercept: float, intercept in the model y = wx + intercept, for 
            unpreprocessed raw data.  
    """
    if use_CV:
        i_fold = cache["i_fold"]
        X_offset = np.array(cache["mean_X_folds"][i_fold], dtype="float64")
        y_offset = cache["mean_y_folds"][i_fold]
        X_scale = np.array(cache["scale_X_folds"][i_fold], dtype="float64")
    else:
        X_offset = np.array(cache["mean_X_train"], dtype="float64")
        y_offset = cache["mean_y_train"]
        X_scale = np.array(cache["scale_X_train"], dtype="float64")

    X_scale = np.where(X_scale == 0.0, 1.0, X_scale)
    w = w / X_scale
    intercept = y_offset - np.dot(X_offset, w)

    return w.tolist(), intercept


def predict(X, w, intercept):
    """Returns the prediction y given X and the linear model.

    Args:
        X: np.ndarray of shape (n_samples, n_features).
        w: np.ndarray of shape (n_features,), slope in the model 
            y = wx + intercept, for unpreprocessed raw data.
        intercept: float, intercept in the model y = wx + intercept, 
            for unpreprocessed raw data.          
    """
    return np.matmul(X, w) + intercept


def squared_error(a, b):
    """Returns sum of squares of two vectors.

    Args:
        a, b: can be both scalar or both vector.
    """
    return sum(np.square(a - b))


def agg_MSE(input, n_samples):
    """Aggregates MSE from local sites, returns the averaged MSE for the whole.

    Args:
        input: COINSTAC dict.
        n_samples: int, number of samples in the testing/validation dataset.
    """
    return (
        sum(site_dict["se_local"] for site, site_dict in input.items())
        / n_samples
    )


def agg_R2(input):
    """Aggregates squared errors from local sites, returns the whole R2-score.

    Args:
        input: COINSTAC dict.

    Raises:
        Exception: if v == 0, R2-score = 1 - u / v.  
    """
    numerator = sum(site_dict["se_local"] for site, site_dict in input.items())
    denominator = sum(
        site_dict["se_denominator_local"] for site, site_dict in input.items()
    )
    if denominator == 0:
        raise Exception("agg_R2: demonimator is 0")
    return 1 - numerator / denominator


def set_rank(w, name_features):
    """Sets the rank of the coefficients in w.

    Args:
        w: float list, len = n_features, slope in the model y = wx + intercept. 
        name_features: str list, len = n_features, name of each feature.

    Returns:
        a list of pair [coefficient value, name_feature] sorted by the absolute 
            value of the coefficients in decreasing order.
    """
    w = np.array(w)
    name_features = np.array(name_features)

    index = np.argsort(np.abs(w))[::-1]

    if w.ndim == 1:
        w = w[:, np.newaxis]
    if name_features.ndim == 1:
        name_features = name_features[:, np.newaxis]

    return np.concatenate((w[index], name_features[index]), axis=1).tolist()
