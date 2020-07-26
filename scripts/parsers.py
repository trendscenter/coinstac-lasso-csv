"""Parsers to extract features X and labels y from raw data.
"""
import os

import numpy as np
import pandas as pd

# def parse(id, cache_dir, base_dir):
#     with open(os.path.join(base_dir + "/data" + str(id)[-1] + ".npy"), "rb") as fp:
#         Xy = np.load(fp)
#     # raise Exception(str(Xy.shape))
#     X_train = Xy[:, :-1]
#     y_train = Xy[:, -1]

#     # n_features = 10
#     X_test = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0],
#                        [1.0, 2.0, 3.0, 7.0, 6.0, 5.0, 3.0, 9.0, 2.0, 9.0]])
#     y_test = np.array([4.0, 9.0])

#     np.save(os.path.join(cache_dir, "X_train.npy"), X_train)
#     np.save(os.path.join(cache_dir, "X_test.npy"), X_test)
#     np.save(os.path.join(cache_dir, "y_train.npy"), y_train)
#     np.save(os.path.join(cache_dir, "y_test.npy"), y_test)    

#     return X_train, X_test, y_train, y_test

def parse_csv(input):
    """Parses csv in inputspec.json, returns features X and labels y.

    Args:
        input (dict): Input of COINSTAC pipeline at each iteration.
        base_dir (str): baseDirectory at each site.

    Returns:
        features_np (ndarray of shape (n_sample, n_feature)): X.
        labels_np (ndarray of shape (n_sample,)): y.
        name_features

    label should be continuous, not categorical
    """
    # process covariates
    raw = input["covariates"][0][1:]
    cols = input["covariates"][0][0] 
    index_name = cols[0]  # e.g., freesurferfile

    features_df = pd.DataFrame.from_records(raw, columns=cols)
    features_df.set_index(index_name, inplace=True)

    # process measurements
    if input["data"]:
        raw = input["data"][0][1:]
        cols = input["data"][0][0] 
        index_name = cols[0]

        measurements_df = pd.DataFrame.from_records(raw, columns=cols)
        measurements_df.set_index(index_name, inplace=True)   

        # merge measurements_df into features_df
        features_df = features_df.merge(
            measurements_df, how="inner", left_index=True, right_index=True
        )

    # convert str to numeric, process categorical and boolean classes
    # Note:
    #     if a categorical class is read in as int or as numerical
    #     string, e.g.,'1', then you should manually set this class
    #     as categorical type before calling pd.get_dummies():
    #         df['A'] = df['A'].astype('category')
    #     Otherwise, pd.to_numeric converts is to int/float.    
    features_df = features_df.apply(pd.to_numeric, errors="ignore")
    features_df = pd.get_dummies(features_df, drop_first=True)  # convert categorical features
    features_df = features_df * 1  # True -> 1, False -> 0
    features_df = features_df.apply(pd.to_numeric, errors="ignore")  # object 0, 1 -> int

    # separate features_df to get features matrix and label matrix
    label_name = input["label"]
    label_df = features_df.pop(label_name)

    # convert to np.ndarray
    features_np = features_df.to_numpy()
    label_np = label_df.to_numpy().flatten()

    # raise Exception(list(features_df.columns))

    return features_np, label_np, list(features_df.columns)
