"""Script run at the remote site.

The remote site:
    0. aggregates local stats to set the mean, scale and sum of squares of each 
        feature of the whole. 
    1. initiates the training, aggregates the local updates to train the model.
    2. aggregates local performance metrics of testing for the whole metrics.  

Raises:
    Exception: If the value of "msg" from sys.stdin is neither empty nor in 
        ["to_agg_mean", "to_agg_scale", "to_init_train", "to_agg_train", 
        "to_agg_test"].
"""
import sys

import numpy as np
import ujson as json
import copy

from common_functions import list_recursive
from remote_ancillary import (
    readin_remote,
    receive_params,
    Stats_agg,
    Stats_agg_CV,
)
from remote_ancillary import agg_scale_sqsum, agg_scale_sqsum_CV
from remote_ancillary import set_lambdas, pick_jth_feature
from linear_model_ancillary import set_w_intercept, agg_MSE, agg_R2, set_rank


def remote_agg_mean(args):
    input, state, cache = readin_remote(args)
    owner = state["owner"] if "owner" in state else "local0"
    receive_params(input[owner], cache)
    preprocess_method = cache["preprocess_method"]
    n_features = cache["n_features"]
    use_CV = cache["use_CV"]
    cache["used_CV"] = False

    # agg stats
    output = {}
    stats = Stats_agg(preprocess_method, n_features)
    stats.agg_stats(input)
    stats.add_output(output)
    stats.add_cache(cache)
    if use_CV:
        n_folds = cache["n_folds"]
        stats_CV = Stats_agg_CV(preprocess_method, n_features, n_folds)
        stats_CV.agg_stats(input)
        stats_CV.add_output(output)
        stats_CV.add_cache(cache)

    result_dict = {"output": output, "cache": cache}
    return json.dumps(result_dict)


def remote_agg_scale(args):
    input, state, cache = readin_remote(args)
    n_features = cache["n_features"]
    use_CV = cache["use_CV"]
    preprocess_method = cache["preprocess_method"]

    output = {}
    scale_X_train, sqsum_X_train = agg_scale_sqsum(
        input, preprocess_method, n_features, cache["n_samples_train"]
    )
    output["mean_X_train"] = cache["mean_X_train"]
    output["mean_y_train"] = cache["mean_y_train"]
    output["scale_X_train"] = scale_X_train
    cache["scale_X_train"] = scale_X_train
    cache["sqsum_X_train"] = sqsum_X_train

    if use_CV:
        n_folds = cache["n_folds"]
        n_samples_folds = cache["n_samples_folds"]
        scale_X_folds, sqsum_X_folds = agg_scale_sqsum_CV(
            input, preprocess_method, n_features, n_samples_folds, n_folds
        )

        output["mean_X_folds"] = cache["mean_X_folds"]
        output["mean_y_folds"] = cache["mean_y_folds"]
        output["scale_X_folds"] = scale_X_folds
        cache["scale_X_folds"] = scale_X_folds
        cache["sqsum_X_folds"] = sqsum_X_folds

    # dicts
    output["msg"] = "to_preprocess"

    result_dict = {"output": output, "cache": cache}
    return json.dumps(result_dict)


def remote_init_train(args):
    input, state, cache = readin_remote(args)
    n_features = cache["n_features"]
    use_CV = cache["use_CV"]

    if not list(list_recursive(cache, "i_lambda")):
        cache["i_lambda"] = 0
        lambdas = set_lambdas(cache["lambdas"], input, cache)
        cache["lambdas"] = lambdas
        cache["n_lambdas"] = len(lambdas)

    # init i_fold, i_lambda
    if use_CV:
        if not list(list_recursive(cache, "i_fold")):
            cache["i_fold"] = 0
        if not list(list_recursive(cache, "MSEs_CV")):
            cache["MSEs_CV"] = []
            cache["convergeds_CV"] = []
            cache["n_iters_CV"] = []
    else:
        if not list(list_recursive(cache, "ws")):
            cache["ws_for_scaled_data"] = []
            cache["ws"] = []
            cache["intercepts"] = []
            cache["R2s"] = []
            cache["MSEs"] = []
            cache["w_ranks"] = []
            cache["convergeds"] = []
            cache["n_iters"] = []

    # init
    i_iter = 0
    i_feature = 0
    w = np.zeros(n_features).tolist()
    jj = pick_jth_feature(cache["selection"], n_features, i_feature)
    dw = 0.0

    # dicts
    output = {"msg": "to_train", "w": w, "jj": jj}
    if use_CV:
        output["i_fold"] = cache["i_fold"]

    cache["i_iter"] = i_iter
    cache["i_feature"] = i_feature
    cache["w"] = w
    cache["jj"] = jj
    cache["dw"] = dw

    result_dict = {"output": output, "cache": cache}
    return json.dumps(result_dict)


def remote_agg_train(args):
    input, state, cache = readin_remote(args)
    use_CV = cache["use_CV"]

    n_features = cache["n_features"]
    max_iter = cache["max_iter"]
    tol = cache["tol"]
    positive = cache["positive"]
    selection = cache["selection"]

    lambdas = cache["lambdas"]
    i_lambda = cache["i_lambda"]
    if use_CV:
        i_fold = cache["i_fold"]
        lambda_ = lambdas[i_lambda] * cache["n_samples_folds"][i_fold]
    else:
        lambda_ = lambdas[i_lambda] * cache["n_samples_train"]

    w = cache["w"]
    jj = cache["jj"]
    i_iter = cache["i_iter"]
    i_feature = cache["i_feature"]
    dw = float(cache["dw"])  # float

    if use_CV:
        sqsum_X = np.array(cache["sqsum_X_folds"][i_fold], dtype="float64")
    else:
        sqsum_X = np.array(cache["sqsum_X_train"], dtype="float64")

    # agg c_jj
    c_jj = sum(site_dict["c_jj_local"] for site, site_dict in input.items())

    # update w_jj
    w_prev = w[jj]  # store previous value
    if positive and c_jj < 0:
        w[jj] = 0.0
    else:  # soft function
        if sqsum_X[jj] != 0:
            w[jj] = np.sign(c_jj) * max(abs(c_jj) - lambda_, 0) / sqsum_X[jj]

    # update dw
    dw += abs(w_prev - w[jj])

    # update i_feature
    i_feature += 1
    if i_feature == n_features:
        if dw < tol:  # reached tolerance, terminate training
            w_for_scaled_data = copy.deepcopy(w)
            w, intercept = set_w_intercept(w, use_CV, cache)
            output = {"msg": "to_test", "w": w, "intercept": intercept}
            if use_CV:
                output["i_fold"] = cache["i_fold"]
            else:
                output["mean_y_test"] = cache["mean_y_test"]  # for r2_score

            cache["w_for_scaled_data"] = w_for_scaled_data
            cache["w"] = w
            cache["intercept"] = intercept
            cache["converged"] = True
            cache["n_iter"] = i_iter + 1
            result_dict = {"output": output, "cache": cache}
            return json.dumps(result_dict)
        else:  # update i_iter
            i_iter += 1
            if i_iter >= max_iter:  # reached max_iter, terminate training
                w_for_scaled_data = copy.deepcopy(w)
                w, intercept = set_w_intercept(w, use_CV, cache)
                output = {"msg": "to_test", "w": w, "intercept": intercept}
                if use_CV:
                    output["i_fold"] = cache["i_fold"]
                else:
                    output["mean_y_test"] = cache["mean_y_test"]  # for r2_score

                cache["w_for_scaled_data"] = w_for_scaled_data
                cache["w"] = w
                cache["intercept"] = intercept
                cache["converged"] = False
                cache["n_iter"] = max_iter
                result_dict = {"output": output, "cache": cache}
                return json.dumps(result_dict)
            else:  # reset for next iter
                i_feature = 0
                dw = 0.0

    #  update jj
    jj = pick_jth_feature(selection, n_features, i_feature)

    # dicts
    output = {"msg": "to_train", "w": w, "jj": jj}
    if use_CV:
        output["i_fold"] = cache["i_fold"]

    cache["i_iter"] = i_iter
    cache["i_feature"] = i_feature
    cache["w"] = w
    cache["jj"] = jj
    cache["dw"] = dw

    result_dict = {"output": output, "cache": cache}
    return json.dumps(result_dict)


def remote_agg_test(args):
    input, state, cache = readin_remote(args)
    use_CV = cache["use_CV"]
    i_lambda = cache["i_lambda"]
    n_lambdas = cache["n_lambdas"]

    if use_CV:
        # agg MSE, loop over i_lambda and i_fold
        n_folds = cache["n_folds"]
        i_fold = cache["i_fold"]
        n_samples = cache["n_samples_valids"][i_fold]
        cache["MSEs_CV"].append(agg_MSE(input=input, n_samples=n_samples))
        cache["convergeds_CV"].append(cache["converged"])
        cache["n_iters_CV"].append(cache["n_iter"])

        i_lambda += 1  # update i_lambda
        if i_lambda == n_lambdas:
            i_fold += 1  # update i_fold
            if i_fold == n_folds:  # CV completed, train at best_lambda
                MSEs_CV_2D = np.reshape(
                    np.array(cache["MSEs_CV"]), (n_lambdas, n_folds), order="F"
                )
                cache["MSEs_CV"] = MSEs_CV_2D.tolist()
                index = np.argmin(np.sum(MSEs_CV_2D, axis=1))
                best_lambda = cache["lambdas"][index]
                cache["convergeds_CV"] = np.reshape(
                    np.array(cache["convergeds_CV"]),
                    (n_lambdas, n_folds),
                    order="F",
                ).tolist()
                cache["n_iters_CV"] = np.reshape(
                    np.array(cache["n_iters_CV"]),
                    (n_lambdas, n_folds),
                    order="F",
                ).tolist()
                cache["use_CV"] = False
                cache["used_CV"] = True
                cache["lambdas_CV"] = cache["lambdas"]
                cache["i_lambda"] = 0
                cache["n_lambdas"] = 1
                cache["lambdas"] = [best_lambda]
            else:  # train next CV fold
                cache["i_fold"] = i_fold
                cache["i_lambda"] = 0  # reset
        else:  # train at next lambda
            cache["i_lambda"] = i_lambda
    else:
        # agg MSE and R2 score, loop over i_lambda
        cache["ws_for_scaled_data"].append(cache["w_for_scaled_data"])
        cache["ws"].append(cache["w"])
        cache["intercepts"].append(cache["intercept"])
        cache["MSEs"].append(
            agg_MSE(input=input, n_samples=cache["n_samples_test"])
        )
        cache["R2s"].append(agg_R2(input=input))
        cache["w_ranks"].append(
            set_rank(cache["w_for_scaled_data"], cache["name_features"])
        )
        cache["convergeds"].append(cache["converged"])
        cache["n_iters"].append(cache["n_iter"])

        i_lambda += 1  # update i_lambda
        if i_lambda == n_lambdas:  # program ending!!!
            output = {}
            output["label"] = [cache["label"], "string"]
            output["train_split_local"] = [cache["train_split_local"], "array"]
            output["train_split_owner"] = [cache["train_split_owner"], "array"]
            output["train_index"] = [cache["train_index"], "array"]
            output["test_index"] = [cache["test_index"], "array"]
            output["preprocess_method"] = [cache["preprocess_method"], "string"]
            output["max_iter"] = [cache["max_iter"], "number"]
            output["tol"] = [cache["tol"], "number"]
            output["positive"] = [cache["positive"], "boolean"]
            output["selection"] = [cache["selection"], "string"]
            output["eps"] = [cache["eps"], "number"]
            output["n_lambdas"] = [cache["n_lambdas"], "number"]
            output["use_CV"] = [cache["used_CV"], "boolean"]
            output["ws_for_scaled_data"] = [
                cache["ws_for_scaled_data"],
                "array",
            ]
            output["ws"] = [cache["ws"], "array"]
            output["intercepts"] = [cache["intercepts"], "array"]
            output["MSEs"] = [cache["MSEs"], "array"]
            output["R2s"] = [cache["R2s"], "array"]
            output["w_ranks"] = [cache["w_ranks"], "array"]
            output["convergeds"] = [cache["convergeds"], "array"]
            output["n_iters"] = [cache["n_iters"], "array"]

            if cache["used_CV"]:
                output["n_folds"] = [cache["n_folds"], "number"]
                output["fold_indices"] = [cache["fold_indices"], "array"]
                output["valid_indices"] = [cache["valid_indices"], "array"]
                output["lambdas_CV"] = [cache["lambdas_CV"], "array"]
                output["MSEs_CV (n_lambdas, n_folds)"] = [
                    cache["MSEs_CV"],
                    "array",
                ]
                output["best_lambda"] = [cache["lambdas"][0], "number"]
                output["convergeds_CV (n_lambdas, n_folds)"] = [
                    cache["convergeds_CV"],
                    "array",
                ]
                output["n_iters_CV (n_lambdas, n_folds)"] = [
                    cache["n_iters_CV"],
                    "array",
                ]
            else:
                output["lambdas"] = [cache["lambdas"], "array"]

            result_dict = {"output": output, "success": True}
            return json.dumps(result_dict)
        else:  # train at next lambda
            cache["i_lambda"] = i_lambda

    args = {"input": input, "state": state, "cache": cache}
    return remote_init_train(args)


if __name__ == "__main__":
    parsed_args = json.loads(sys.stdin.read())
    msg = list(list_recursive(parsed_args, "msg"))

    if "to_agg_mean" in msg:
        result_dict = remote_agg_mean(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_agg_scale" in msg:
        result_dict = remote_agg_scale(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_init_train" in msg:
        result_dict = remote_init_train(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_agg_train" in msg:
        result_dict = remote_agg_train(parsed_args)
        sys.stdout.write(result_dict)
    elif "to_agg_test" in msg:
        result_dict = remote_agg_test(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception("Error occurred at Remote")
