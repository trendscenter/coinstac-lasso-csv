"""Parsers to extract features X and label y from the data in inputspec.json.
"""
import os

import numpy as np
import pandas as pd

def parse_csv(input):
    """Parses CSV-style data in inputspec.json.

    For Lasso, label y should be continuous.

    Args:
        input: COINSTAC input dict.

    Returns:
        features_np: np.ndarray of shape (n_sample, n_feature), X.
        labels_np: np.ndarray of shape (n_sample,), y.
        name_features: str list, name of each feature. 
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

    return features_np, label_np, list(features_df.columns)
