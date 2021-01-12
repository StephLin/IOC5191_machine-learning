# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 02-2: Online Learning
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse

import numpy as np

# -------------------------------- Test cases -------------------------------- #


def load_testcases(filename: str) -> np.ndarray:
    dataset = []
    with open(filename, 'r') as fail:
        for line in fail:
            series = line.strip()
            n = len(series)
            success = np.sum(list(map(int, series)), dtype=int)
            fail = n - success
            dataset.append([series, success, fail])
    return dataset


# ----------------------- Beta-Binomial conjugate prior ---------------------- #


def calc_comb(n, k):
    comb = np.math.factorial(n)
    comb /= np.math.factorial(k)
    comb /= np.math.factorial(n - k)
    return comb


def binomial_likelihood(success: int, fail: int):
    p = success / (success + fail)
    comb = calc_comb(success + fail, success)
    return comb * p**success * (1 - p)**fail


def beta_binomial_conjugate_prior(alpha: int, beta: int, success: int,
                                  fail: int):
    return [success + alpha, fail + beta]


# ------------------------------- Main function ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='filename of test cases')
    parser.add_argument('alpha',
                        type=int,
                        help=('alpha parameter for beta distribution '
                              '(integer only)'))
    parser.add_argument('beta',
                        type=int,
                        help=('beta parameter for beta distribution '
                              '(integer only)'))
    args = parser.parse_args()

    testcases = load_testcases(args.filename)
    alpha = args.alpha
    beta = args.beta

    for idx, testcase in enumerate(testcases):
        series, success, fail = testcase
        print('case %d: %s' % (idx + 1, series))
        print('Likelihood: %.5f' % binomial_likelihood(success, fail))
        print('Beta prior:      a = %2d b = %2d' % (alpha, beta))
        alpha, beta = beta_binomial_conjugate_prior(alpha, beta, success, fail)
        print('Beta posteriori: a = %2d b = %2d\n' % (alpha, beta))