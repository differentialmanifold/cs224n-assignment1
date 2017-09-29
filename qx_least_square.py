#!/usr/bin/env python

import numpy as np

from q2_gradcheck import gradcheck_naive

from q3_sgd import sgd


def least_square(X, Y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))


def least_square_num(X, Y, params):
    assert X.shape[1] == len(params)

    cost = ((Y - np.dot(X, params)) ** 2).sum() / 2
    grad = np.dot(X.T, (np.dot(X, params) - Y))

    return cost, grad


def least_square_gd(dataset_x, dataset_y):
    np.random.seed(47)
    x0 = np.random.randn(dataset_x.shape[1])
    x = sgd(lambda params: least_square_num(dataset_x, dataset_y, params), x0, 0.01, 500000)
    return x


def sanity_check():
    print "run least square test..."

    dataset_x = np.array([[3.6, 1], [3.7, 1], [3.8, 1], [3.9, 1], [4.0, 1], [4.1, 1], [4.2, 1]])
    dataset_y = np.array([1.00, 0.90, 0.90, 0.81, 0.60, 0.56, 0.35])

    test = least_square(dataset_x, dataset_y)

    ans = np.array([-1.04642857, 4.8125])
    assert np.allclose(test, ans, rtol=1e-05, atol=1e-06)

    params = np.random.randn(dataset_x.shape[1])
    gradcheck_naive(lambda input_x: least_square_num(dataset_x, dataset_y, input_x), params)

    print least_square_gd(dataset_x, dataset_y)

    print "pass least square test"


if __name__ == "__main__":
    sanity_check()
