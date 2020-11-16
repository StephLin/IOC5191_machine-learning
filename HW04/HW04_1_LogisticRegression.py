# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 04-1: Logistic Regression
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------ Box-Muller Method for Normal Distribution Generator ----------- #


def box_muller_method(u, v):
    assert np.all((u > 0) & (u < 1))
    assert np.all((v > 0) & (v < 1))

    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)


def standard_normal_distribution_generator(number_of_samples: int):
    u = np.random.uniform(0, 1, number_of_samples)
    v = np.random.uniform(0, 1, number_of_samples)
    dots = box_muller_method(u, v)
    return dots


def normal_distribution_generator(mean: float, var: float,
                                  number_of_samples: int):
    assert var > 0
    sigma = np.sqrt(var)
    dots = standard_normal_distribution_generator(number_of_samples)
    return dots * sigma + mean


# ----------------------------- Utility functions ---------------------------- #


def matrix_is_invertible(mat):
    return np.linalg.matrix_rank(mat) == mat.shape[0]


# ---------------------------- Logistic Regression --------------------------- #


def regression_form(cluster1: np.ndarray, cluster2: np.ndarray):
    x = np.concatenate((cluster1, cluster2))
    x_ = np.ones((x.shape[0], x.shape[1] + 1))
    x_[:, 1:] = x
    y = np.array([0] * cluster1.shape[0] + [1] * cluster2.shape[0])
    return x_, y


def logistic_function(val):
    return 1 / (1 + np.exp(-val))


def gradient(x, y, w):
    vector = logistic_function(x @ w) - y
    return x.T @ vector


def hessian(x, y, w):
    wx = x @ w
    d = np.diag(np.exp(-wx) * logistic_function(wx)**2)
    return x.T @ d @ x


def predict(data, w):
    x = np.ones((data.shape[0], data.shape[1] + 1))
    x[:, 1:] = data
    perceptrum = logistic_function(x @ w)
    return np.where(perceptrum >= 0.5, 1, 0)


def logistic_regression_newton(cluster1: np.ndarray,
                               cluster2: np.ndarray,
                               init_weights: np.ndarray,
                               max_iter: int = 100,
                               tol: float = 1e-3,
                               step_size=0.05):
    x, y = regression_form(cluster1, cluster2)

    w = init_weights
    grad = gradient(x, y, w)
    n_iter = 0
    # newton's method
    while np.linalg.norm(grad) > tol and n_iter < max_iter:
        hess = hessian(x, y, w)
        if matrix_is_invertible(hess):
            w -= np.linalg.inv(hess) @ grad * step_size
        else:
            w -= grad
        grad = gradient(x, y, w)
        n_iter += 1
    return w


def logistic_regression_gradient(cluster1: np.ndarray,
                                 cluster2: np.ndarray,
                                 init_weights: np.ndarray,
                                 max_iter: int = 100,
                                 tol: float = 1e-3,
                                 step_size: float = 0.05):
    x, y = regression_form(cluster1, cluster2)

    w = init_weights
    grad = gradient(x, y, w)
    n_iter = 0
    # gradient descent
    while np.linalg.norm(grad) > tol and n_iter < max_iter:
        w -= grad * step_size
        grad = gradient(x, y, w)
        n_iter += 1
    return w


# ------------------------------- Visualization ------------------------------ #


def predict_and_visualize(cluster1, cluster2, w, ax, title):
    predict1 = predict(cluster1, w)
    predict2 = predict(cluster2, w)
    predicts = predict(np.concatenate((cluster1, cluster2)), w)
    cluster1_ = clusters[np.argwhere(predicts == 0).flatten(), :]
    cluster2_ = clusters[np.argwhere(predicts == 1).flatten(), :]
    print('\n--------------------------------------------------------------\n')
    print('%s:\n' % title)
    print('w:', w, end='\n\n')

    tp = predict1.shape[0] - np.sum(predict1)
    fn = np.sum(predict1)
    fp = predict2.shape[0] - np.sum(predict2)
    tn = np.sum(predict2)
    print('Confusion matrix:')
    print('             Predict cluster 1  Predict cluster 2')
    print('Is cluster 1 %17d  %17d' % (tp, fn))
    print('Is cluster 2 %17d  %17d\n' % (fp, tn))

    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    print('Sensitivity (Successfully predict cluster 1): %.5f' % sensitivity)
    print('Specificity (Successfully predict cluster 2): %.5f' % specificity)

    ax.set_title(title)
    ax.scatter(cluster1_[:, 0], cluster1_[:, 1], color='red')
    ax.scatter(cluster2_[:, 0], cluster2_[:, 1], color='blue')


# ------------------------------- Main function ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='number of data points')
    parser.add_argument('mx1', type=float, help='mean of x1')
    parser.add_argument('vx1', type=float, help='variance of x1')
    parser.add_argument('my1', type=float, help='mean of y1')
    parser.add_argument('vy1', type=float, help='variance of y1')
    parser.add_argument('mx2', type=float, help='mean of x2')
    parser.add_argument('vx2', type=float, help='variance of x2')
    parser.add_argument('my2', type=float, help='mean of y2')
    parser.add_argument('vy2', type=float, help='variance of y2')

    args = parser.parse_args()

    assert args.N > 0
    assert args.vx1 > 0
    assert args.vy1 > 0
    assert args.vx2 > 0
    assert args.vy2 > 0

    cluster1_xs = normal_distribution_generator(args.mx1, args.vx1, args.N)
    cluster1_ys = normal_distribution_generator(args.my1, args.vy1, args.N)
    cluster1 = np.array([cluster1_xs, cluster1_ys]).T

    cluster2_xs = normal_distribution_generator(args.mx2, args.vx2, args.N)
    cluster2_ys = normal_distribution_generator(args.my2, args.vy2, args.N)
    cluster2 = np.array([cluster2_xs, cluster2_ys]).T

    clusters = np.concatenate((cluster1, cluster2))

    # logistic regression by gradient descent
    w_grad = logistic_regression_gradient(cluster1, cluster2, np.ones(3))

    # logistic regression by newton's method
    w_newton = logistic_regression_newton(cluster1, cluster2, np.ones(3))

    # visualization
    fig, (ax_truth, ax_grad, ax_newton) = plt.subplots(1, 3)
    ax_truth.set_title('Ground truth')
    ax_truth.scatter(cluster1_xs, cluster1_ys, color='red')
    ax_truth.scatter(cluster2_xs, cluster2_ys, color='blue')

    predict_and_visualize(cluster1, cluster2, w_grad, ax_grad,
                          'Gradient descent')
    predict_and_visualize(cluster1, cluster2, w_newton, ax_newton,
                          'Newton\'s method')

    plt.show()
