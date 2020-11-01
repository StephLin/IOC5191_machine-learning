# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 03-1: Random Data Generator
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import numpy as np

import matplotlib.pyplot as plt

MAX_ITERATION = 1000
CONVERGENCE = 1e-8

# ------------ Box-Muller Method for Normal Distribution Generator ----------- #


def box_muller_method(u, v):
    assert np.all((u > 0) & (u < 1))
    assert np.all((v > 0) & (v < 1))

    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)


def standard_normal_distribution_generator(number_of_samples):
    u = np.random.uniform(0, 1, number_of_samples)
    v = np.random.uniform(0, 1, number_of_samples)
    dots = box_muller_method(u, v)
    return dots


def polynomial_model_data_generator(n, weights, sigma, number_of_samples):
    assert sigma > 0

    x = np.random.uniform(-1, 1, number_of_samples)
    phi_x = np.array([x**i for i in range(n)])
    dots = standard_normal_distribution_generator(number_of_samples)
    y = weights @ phi_x + dots * sigma
    return x, y


# ------------------------- Baysian Linear Regression ------------------------ #


def bayesian_gaussian(capital_x, y, mean, info, a):
    posterior_info = a * capital_x.T @ capital_x + info
    posterior_mean = a * capital_x.T @ y + info @ mean
    posterior_mean = np.linalg.inv(posterior_info) @ posterior_mean
    return posterior_mean, posterior_info


# ----------------------------- Utility function ----------------------------- #


def visualization(mean, n, var, a, x_dots, y_dots, title=''):
    xs = np.linspace(-2, 2, 100)
    capital_xs = np.array([x**np.arange(n) for x in xs])
    ys = capital_xs @ mean
    sigmas = a + np.diag(capital_xs @ var @ capital_xs.T)
    upper_ys = (capital_xs @ mean + sigmas).flatten()
    lower_ys = (capital_xs @ mean - sigmas).flatten()

    plt.figure(title)
    plt.title(title)
    plt.xlim(-2, 2)
    plt.ylim(-25, 25)

    plt.plot(xs, ys, color='black')
    plt.plot(xs, upper_ys, color='red')
    plt.plot(xs, lower_ys, color='red')
    if len(x_dots) > 0:
        plt.scatter(x_dots, y_dots, color='C0')


# ------------------------------- Main function ------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('b',
                        type=int,
                        help='the precision for the initial prior')
    parser.add_argument('n', type=int, help='degree of the polynoimal space')
    parser.add_argument('var',
                        type=float,
                        help=('variance of the normal distribution '
                              '(the value is shared as `a` for 1.b)'))
    parser.add_argument('--weights', nargs='+')
    args = parser.parse_args()

    assert args.b > 0
    assert args.n > 0
    assert args.var > 0
    assert type(args.weights) == list and len(args.weights) == args.n

    a = args.var
    sigma = np.sqrt(args.var)
    n = args.n
    weights = np.array(list(map(float, args.weights)))

    mean = np.zeros(n)
    info = np.diag([1 / args.b] * n)

    n_iter = 0
    x_dots = []
    y_dots = []

    visualization(weights, n, np.zeros((n, n)), a, x_dots, y_dots,
                  'Ground truth')

    while n_iter < MAX_ITERATION:
        n_iter += 1
        x, y = polynomial_model_data_generator(n, weights, sigma, 1)
        x_dots.append(x)
        y_dots.append(y[0])
        print('(%d) Add data point (%.5f, %.5f):\n' % (n_iter, x, y))

        capital_x = x**np.arange(n)[:, np.newaxis].T
        p_mean, p_info = bayesian_gaussian(capital_x, y, mean, info, a)
        p_var = np.linalg.inv(p_info)

        mean_diff = (p_mean - mean)**2
        info_diff = (p_info - info)**2

        print('Posterior mean:\n', p_mean, '\n')
        print('Posterior variance:\n', p_var, '\n')

        predict_mean = p_mean @ capital_x.T
        predict_var = a + capital_x @ p_var @ capital_x.T
        print('Predictive distribution ~ N(%.5f, %.5f)' %
              (predict_mean, predict_var))

        mean = p_mean
        info = p_info

        if n_iter in [10, 50]:
            visualization(mean, n, p_var, a, x_dots, y_dots,
                          'After %d incomes' % n_iter)

        if np.sum(mean_diff) < CONVERGENCE:
            break

        print('--------------------------------------------------------------')

    visualization(mean, n, np.linalg.inv(info), a, x_dots, y_dots,
                  'Predict result')

    plt.show()
