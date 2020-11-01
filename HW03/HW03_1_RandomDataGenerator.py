# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 03-1: Random Data Generator
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import numpy as np

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
    return y


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
    parser.add_argument('n', type=int, help='degree of the polynoimal space')
    parser.add_argument('--weights', nargs='+')
    args = parser.parse_args()

    assert args.var > 0
    assert args.n > 0
    assert type(args.weights) == list and len(args.weights) == args.n

    mu = args.mean
    sigma = np.sqrt(args.var)
    n = args.n
    weights = np.array(list(map(float, args.weights)))

    # part A
    dots = standard_normal_distribution_generator(1)
    print('a.', dots[0] * sigma + mu)

    # part B
    y = polynomial_model_data_generator(n, weights, sigma, 1)
    print('b.', y[0])
