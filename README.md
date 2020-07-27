# coinstac-lasso-csv
COINSTAC implementation for decentralized multi-shot Lasso regression for generic CSV files.

Run a testing case: `./run.sh`


### 1. Background info
- Linear model: y = wx + w0.
- L1-regularized objective: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + lambda * ||w||_1. 
- Optimization algorithm: coordinate descent (shooting algorithm) [[1]](#1), [[2]](#2).

### 2. Two functionalities
1. Pick the best L1-regularizer lambda by cross validation (CV) and then train & test at that lambda. Set `use_CV = true`.
    - The best lambda is the one generating the smallest mean squared errors (MSE) averaged over the folds.
    - Note that CV decides the best lambda based on predictive accuracy, thus it tends to be a small one and may not be the one in the "true" model [[2]](#2). 
    - Sample input: [inputspec_CV.json](test/inputspec_CV.json)
    - Sample output: [output_CV.json](output_CV.json)

    <img src="https://github.com/trendscenter/coinstac-lasso-csv/blob/master/output_CV.png?raw=true" width=400 height=650>    
   
2. Train & test at a list of lambdas. Set `use_CV = false`.  
    - Sample input: [inputspec_nonCV.json](test/inputspec_nonCV.json)
    - Sample output: [output_nonCV.json](output_nonCV.json) 

    <img src="https://github.com/trendscenter/coinstac-lasso-csv/blob/master/output_nonCV.png?raw=true" width=450 height=900> 
      
    - The regularization path along the list of lambdas can be obtained from `ws_for_scaled_data` in the output:
    <img src="https://github.com/trendscenter/coinstac-lasso-csv/blob/master/regularization_paths.png?raw=true"> 

### 3. Input
- `label`: any name of a continuous-valued feature in CSV files.

- train/test split:
    - `train_split_local/owner`: proportion of data used for training at all local sites but the owner site/at the owner site. If <= 0, all used for testing. If a float within (0, 1), proportion of data used for training. If >= 1, all used for training.
    - There should be both training and testing data. No testing (i.e., `train_split_local >= 1` and `train_split_owner >= 1`) or no training data (i.e., `train_split_local <= 0` and `train_split_owner <= 0`) will trigger a ValueError.
    - To have repeatable runs, you can set `shuffle = false` or you can set `shuffle = true` and `random_state = an integer`. Otherwise, you can set `shuffle = true` and `random_state = null`. It's the same for the settings in CV, `shuffle_CV` and `random_state_CV`.

- preprocess method: four options are provided to preprocess the training data X (n_samples, n_features) and y (n_samples,) before training. The operations on X are column-wise.
    - 'none': data unpreprocessed.
    - 'center': X = X - X_mean, y = y - y_mean.
    - 'normalize': X = (X - X_mean) / X_L2_norm, where X_L2_norm is calculated from (X - X_mean), y = y - y_mean. 
    - 'standardize': X = (X - X_mean) / X_std, y = y - y_mean.

- cooridnate descent:
    - To only retain positive coefficents and shrink all negative coefficients to 0, set `positive = true`.
    - Two options are provided for the order of updating coefficients. To update them sequentially in each iteration, set `selection = 'cyclic'`. To update them randomly, set `selection = 'random'`.

- lambdas:
    - If `use_CV = true`, CV will be conducted along the `lambdas` and the best lambda will be chosen from the `lambdas`. If `use_CV = false`, a model will be computed at each `lambdas`.
    - There are two ways to input lambdas. One is to directly input a list of values in `lambdas`. The other one is to set `lamdas` empty as `lambdas = []` and set values for `eps` and `n_lambdas`. This program will automatically calculate lambda_max (i.e., if lambda >= lambda_max, w = 0), then lambda_min = `eps` * lambda_max, and finally set a list of lambdas of length `n_lambdas` spaced evenly on a log scale. 

### 4. Output
- w is outputed in two forms. `ws_for_scaled_data` contains the w's for preprocessed data. `ws` contains the w's for unpreprocessed data, i.e., the raw input data. 
- `w_ranks`, showing the (coefficient, feature_name) pair sorted by the coefficient's abosulute value, is based on `ws_for_scaled_data`.  


### 5. Code
The following diagram shows the framework implemented in [local.py](scripts/local.py) and [remote.py](scripts/remote.py).

<img src="https://github.com/trendscenter/coinstac-lasso-csv/blob/master/decentralized_lasso.png?raw=true"> 


### References
<a id="1">[1]</a> Fu, W. J. (1998). Penalized regressions: the bridge versus the lasso. Journal of computational and graphical statistics, 7(3), 397-416.

<a id="2">[2]</a> Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

