import numpy as np
from scipy import stats as st
from simulation import data_simulation

def std_Z(X, W, alpha, qW_hat):
    return np.sqrt(np.mean(W**2 * (alpha - (X<=qW_hat))**2))

def confidence_interval(X, W, alpha, eta):
    qW_hat = np.quantile(a=X, q=alpha, weights=W, method='inverted_cdf')  # Weighted quantile estimator
    c_eta = st.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    c = c_eta * std_Z(X, W, alpha, qW_hat) / np.mean(W)
    alpha_left = np.maximum(0, alpha-(c/np.sqrt(n)))
    alpha_right = np.minimum(alpha + (c/np.sqrt(n)), 1)
    ci_left = np.quantile(a=X, q=alpha_left, weights=W, method='inverted_cdf')
    ci_right = np.quantile(a=X, q=alpha_right, weights=W, method='inverted_cdf')
    return ci_left, ci_right, qW_hat

def fit_ci(scenario, n_replications, n_samples, theta, alpha, eta, qW_real):
    np.random.seed(scenario)
    dict_result = {'ci_left': [], 'ci_right': [], 'ci_width': [], 'coverage': []}
    for rep in range(n_replications):
        X_samples, W_samples = data_simulation(scenario=scenario, n=n_samples, theta=theta, seed=None)
        ci_left, ci_right, qW_hat = confidence_interval(X_samples, W_samples, alpha, eta)
        dict_result['ci_left'].append(ci_left)
        dict_result['ci_right'].append(ci_right)
        dict_result['ci_width'].append(ci_right - ci_left)
        dict_result['coverage'].append((ci_left <= qW_real) & (qW_real <= ci_right))

    return dict_result