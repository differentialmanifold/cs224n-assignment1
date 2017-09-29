#!/usr/bin/env python

import numpy as np

from q2_gradcheck import gradcheck_naive

from q3_sgd import sgd
from qx_least_square import least_square


def matrix_factorization(matrix, params):
    n, m = matrix.shape
    first_factor = params[:n, :]
    second_factor = params[n:, :]

    matrix_e = matrix - np.dot(first_factor, second_factor.T)

    cost = np.sum(np.square(matrix_e)) / 2

    first_grad = -np.dot(matrix_e, second_factor)
    second_grad = -np.dot(matrix_e.T, first_factor)
    grad = np.vstack((first_grad, second_grad))

    return cost, grad


def matrix_factorization_gd(matrix):
    col_len = 3
    n, m = matrix.shape
    np.random.seed(47)
    x0 = np.random.randn(n + m, col_len)

    x = sgd(lambda params: matrix_factorization(matrix, params), x0, 0.01, 1000)
    return x


def alternative_least_square(matrix, first_factor, second_factor):
    first_factor = np.apply_along_axis(lambda pred: least_square(second_factor, pred), 1, matrix)
    second_factor = np.apply_along_axis(lambda pred: least_square(first_factor, pred), 0, matrix).T

    return first_factor, second_factor


def als_itr(matrix, num):
    col_len = 3
    n, m = matrix.shape
    np.random.seed(47)
    first_factor = np.random.randn(n, col_len)
    second_factor = np.random.randn(m, col_len)

    for i in range(num):
        first_factor, second_factor = alternative_least_square(matrix, first_factor, second_factor)
        params = np.vstack((first_factor, second_factor))
        cost, grad = matrix_factorization(matrix, params)
        print i, cost
        if cost < 1e-6:
            break

    print first_factor
    print second_factor
    print np.dot(first_factor, second_factor.T)


def sanity_check():
    print "run matrix factorization test..."

    matrix = np.arange(1, 13).reshape(3, 4)
    params = np.random.randn(7, 3)

    # gradcheck_naive(lambda input_x: matrix_factorization(matrix, input_x), params)

    # x = matrix_factorization_gd(matrix)
    # n, m = matrix.shape
    # matrix_pred = np.dot(x[:n, :], x[n:, :].T)
    #
    # print(matrix_pred)

    als_itr(matrix, 100)

    print "pass matrix factorization test"


if __name__ == "__main__":
    sanity_check()
