# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 03-2: Sequential Estimator
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import numpy as np

NUMBER_OF_SAMPLES = 3000
CONVERGENCE = 0.05

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


def normal_distribution_generator(mean, variance, number_of_samples):
    assert variance > 0
    dots = standard_normal_distribution_generator(number_of_samples)
    sigma = np.sqrt(variance)
    return dots * sigma + mean


# ----------------------------- Sequential Update ---------------------------- #


def welford_online_algorithm(n, mean, var, dot):
    assert n >= 2
    new_mean = mean + (dot - mean) / n
    new_var = var + (dot - mean)**2 / n - var / (n - 1)
    return new_mean, new_var


# ------------------------------- Main function ------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mean',
                        type=float,
                        help='mean of the normal distribution')
    parser.add_argument('var',
                        type=float,
                        help=('variance of the normal distribution '
                              '(the value is shared as `a` for 1.b)'))
    args = parser.parse_args()
    assert args.var > 0

    source_sigma = np.sqrt(args.var)

    print('Data point source function: N(%.1f, %.1f)' % (args.mean, args.var))
    print()

    n = 1
    mean = normal_distribution_generator(args.mean, args.var, 1)[0]
    var = 0
    while True:
        n += 1
        dot = normal_distribution_generator(args.mean, args.var, 1)[0]
        print('Add data point: %.5f' % dot)

        new_mean, new_var = welford_online_algorithm(n, mean, var, dot)
        mean_diff = np.abs(mean - new_mean)
        var_diff = np.abs(var - new_var)

        mean = new_mean
        var = new_var
        print('Mean = %.5f   Variance = %.5f' % (mean, var))

        if mean_diff < CONVERGENCE and var_diff < CONVERGENCE:
            break
