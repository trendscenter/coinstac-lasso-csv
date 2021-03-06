"""Script run at the owner site and the other local sites.

Each local site: 
    0. calculates the local stats, sends it to the remote site.
    1. preprocesses the local training data based on the aggregated stats sent 
        by the remote site.
    2. calculates the update based on the local training dataset, sends it to 
        the remote site.
    3. tests on the local testing dataset using the model sent by the remote 
        site, sends performance metrics to the remote site. 

Raises:
    Exception: If the value of "msg" from sys.stdin is neither empty nor in 
        ["to_sqsum", "to_preprocess", "to_train", "to_test"].
"""
import os
import sys

import numpy as np
import ujson as json

from local_ancillary import (
    readin,
    split_save_train_test,
    split_folds_save_valid,
    Stats,
    Stats_CV,
)
from local_ancillary import preprocess_save_calXy, preprocess_save_CV
from linear_model_ancillary import predict, squared_error
from common_functions import list_recursive
from parsers import parse_csv


def local_stats_sum(args):
    input, state, cache, cache_dir, id, owner = readin(args)
    base_dir = state["baseDirectory"]
    preprocess_method = input["preprocess_method"]
    use_CV = input["use_CV"]
    output = {}

    (X, y, name_features) = parse_csv(input)
    # split and store train / test dataset
    X_train, X_test, y_train, y_test = split_save_train_test(
        input, output, cache, id, owner, cache_dir, X, y
    )
    # split folds for CV
    if use_CV:
        fold_indices, valid_indices = split_folds_save_valid(
            input, output, cache, cache_dir, X_train, y_train
        )

    # cache dict
    cache["preprocess_method"] = preprocess_method
    cache["use_CV"] = use_CV
    cache["n_features"] = X.shape[1]
    if use_CV:
        cache["n_folds"] = input["n_folds"]

    # calculate stats: n_samples, sum
    stats = Stats(cache, preprocess_method)
    stats.cal_stats(X_train=X_train, y_train=y_train, y_test=y_test)
    stats.add_output(output)
    if use_CV:
        stats_CV = Stats_CV(cache, preprocess_method)
        stats_CV.cal_stats(
            X_train=X_train,
            y_train=y_train,
            fold_indices=fold_indices,
            valid_indices=valid_indices,
        )
        stats_CV.add_output(output)

    # output dict
    if id == owner:
        output["msg"] = "to_agg_mean"
        output["label"] = input["label"]
        output["train_split_local"] = input["train_split_local"]
        output["train_split_owner"] = input["train_split_owner"]
        output["preprocess_method"] = preprocess_method
        output["max_iter"] = input["max_iter"]
        output["tol"] = input["tol"]
        output["positive"] = input["positive"]
        output["selection"] = input["selection"]
        output["lambdas"] = input["lambdas"]
        output["eps"] = input["eps"]
        output["n_lambdas"] = input["n_lambdas"]
        output["use_CV"] = use_CV

        if use_CV:
            output["n_folds"] = input["n_folds"]

        output["name_features"] = name_features

    result_dict = {"output": output, "cache": cache}
    return json.dumps(result_dict)


def local_stats_sqsum(args):
    input, state, cache, cache_dir, id, owner = readin(args)
    use_CV = cache["use_CV"]
    n_features = cache["n_features"]
    output = {}

    if cache["train_split"] > 0:  # has train data
        with open(os.path.join(cache_dir, "X_train.npy"), "rb") as fp:
            X_train = np.load(fp)
        # sqsum
        sqsum_X_train = np.sum(
            np.square(X_train - input["mean_X_train"]), axis=0
        ).tolist()
        output["sqsum_X_train_local"] = sqsum_X_train
        if use_CV:
            sqsum_X_folds = []
            for train_index, mean_X in zip(
                cache["fold_indices"], input["mean_X_folds"]
            ):
                sqsum_X_folds.append(
                    np.sum(
                        np.square(X_train[train_index] - mean_X), axis=0
                    ).tolist()
                )
            output["sqsum_X_folds_local"] = sqsum_X_folds

    else:
        output["sqsum_X_train_local"] = np.zeros(n_features).tolist()
        if use_CV:
            n_folds = cache["n_folds"]
            output["sqsum_X_folds_local"] = np.tile(
                np.zeros(n_features), (n_folds, 1)
            ).tolist()

    # output
    if id == owner:
        output["msg"] = "to_agg_scale"

    result_dict = {"output": output}
    return json.dumps(result_dict)


def local_preprocess(args):
    input, state, cache, cache_dir, id, owner = readin(args)
    use_CV = cache["use_CV"]
    n_features = cache["n_features"]
    output = {}

    if cache["train_split"] > 0:  # has train data
        with open(os.path.join(cache_dir, "X_train.npy"), "rb") as fp:
            X_train = np.load(fp)
        with open(os.path.join(cache_dir, "y_train.npy"), "rb") as fp:
            y_train = np.load(fp)

        # preprocess, save new data, calculate Xy
        # based on preprocessed X/y_train for lambda_max
        output["Xy"] = preprocess_save_calXy(input, cache_dir, X_train, y_train)
        if use_CV:
            preprocess_save_CV(
                input, cache_dir, X_train, y_train, cache["fold_indices"]
            )
    else:
        output["Xy"] = np.zeros(n_features).tolist()

    # dicts
    if id == owner:
        output["msg"] = "to_init_train"

    result_dict = {"output": output}
    return json.dumps(result_dict)


def local_train(args):
    input, state, cache, cache_dir, id, owner = readin(args)

    if cache["train_split"] > 0:  # has train data
        w = np.array(input["w"], dtype="float64")
        jj = input["jj"]

        tmp = list(list_recursive(input, "i_fold"))
        if tmp:  # CV ongoing
            i_fold = tmp[0]
            with open(
                os.path.join(cache_dir, "X_fold_" + str(i_fold) + ".npy"), "rb"
            ) as fp:
                X = np.load(fp)
            with open(
                os.path.join(cache_dir, "y_fold_" + str(i_fold) + ".npy"), "rb"
            ) as fp:
                y = np.load(fp)
        else:  # non-CV
            with open(os.path.join(cache_dir, "X_train.npy"), "rb") as fp:
                X = np.load(fp)
            with open(os.path.join(cache_dir, "y_train.npy"), "rb") as fp:
                y = np.load(fp)

        c_jj = np.dot(X[:, jj], (y - np.matmul(X, w) + w[jj] * X[:, jj]))

    else:
        c_jj = 0.0

    output = {"msg": "to_agg_train", "c_jj_local": float(c_jj)}
    result_dict = {"output": output}
    return json.dumps(result_dict)


def local_test(args):
    input, state, cache, cache_dir, id, owner = readin(args)

    w = np.array(input["w"], dtype="float64")
    intercept = input["intercept"]
    output = {}

    tmp = list(list_recursive(input, "i_fold"))
    if tmp:  # CV ongoing
        if cache["train_split"] > 0:  # has train data
            i_fold = tmp[0]
            with open(
                os.path.join(cache_dir, "X_valid_" + str(i_fold) + ".npy"), "rb"
            ) as fp:
                X_test = np.load(fp)
            with open(
                os.path.join(cache_dir, "y_valid_" + str(i_fold) + ".npy"), "rb"
            ) as fp:
                y_test = np.load(fp)

            y_pred = predict(X_test, w, intercept)
            se = squared_error(y_test, y_pred)  # for MSE
            output["se_local"] = se
        else:
            output["se_local"] = 0.0

    else:  # non-CV
        if cache["train_split"] < 1:  # has test data
            with open(os.path.join(cache_dir, "X_test.npy"), "rb") as fp:
                X_test = np.load(fp)
            with open(os.path.join(cache_dir, "y_test.npy"), "rb") as fp:
                y_test = np.load(fp)

            y_pred = predict(X_test, w, intercept)
            se = squared_error(y_test, y_pred)
            se_denominator = squared_error(
                y_test, input["mean_y_test"]
            )  # for R2 score

            output["se_local"] = float(se)
            output["se_denominator_local"] = float(se_denominator)
        else:
            output["se_local"] = 0.0
            output["se_denominator_local"] = 0.0

    if id == owner:
        output["msg"] = "to_agg_test"

    result_dict = {"output": output}
    return json.dumps(result_dict)


if __name__ == "__main__":
    parsed_args = json.loads(sys.stdin.read())
    msg = list(list_recursive(parsed_args, "msg"))

    if not msg:  # iter 0
        result_dict = local_stats_sum(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_sqsum" in msg:
        result_dict = local_stats_sqsum(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_preprocess" in msg:
        result_dict = local_preprocess(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_train" in msg:
        result_dict = local_train(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_test" in msg:
        result_dict = local_test(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception("Error occurred at Local")
