# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 05-1: Gaussian Process
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

INPUT_FILENAME = './input.data'
BETA = 10
SIGMA_INIT = 1
ALPHA_INIT = 1
L_INIT = 1

# ----------------------------- Utility Functions ---------------------------- #


def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(list(map(float, line.split())))
    data = np.array(data, dtype=float)
    return data[:, 0], data[:, 1]


# ----------------------------- Gaussian Process ----------------------------- #


class GaussianProcess(object):
    def __init__(self, x, y, beta, sigma, alpha, l):
        self.x = x
        self.y = y
        self.beta = beta
        self.beta_inv = 1 / beta
        self.sigma = sigma
        self.alpha = alpha
        self.l = l

    # https://peterroelants.github.io/posts/gaussian-process-kernels/#Rational-quadratic-kernel
    @staticmethod
    def kernel_function(xa, xb, sigma, alpha, l):
        val = 1 + (xa - xb)**2 / (2 * alpha * l**2)
        val = sigma**2 * np.power(val, -alpha)
        return val

    @property
    def kernel_matrix(self):
        x = self.x
        xb = np.tile(x, x.shape).reshape(x.shape + (-1, ))
        xa = xb.T
        return self.kernel_function(xa, xb, self.sigma, self.alpha, self.l)

    @property
    def covariance_matrix(self):
        cov = self.kernel_matrix + self.beta_inv * np.eye(self.x.shape[0])
        return cov

    @property
    def inv_covariance_matrix(self):
        return np.linalg.inv(self.covariance_matrix)

    @property
    def negative_log_likelihood(self):
        x, y = self.x, self.y
        cov = self.covariance_matrix
        val = 0.5 * np.log(np.linalg.det(cov))
        val += 0.5 * y @ np.linalg.inv(cov) @ y
        val += x.shape[0] / 2 * np.log(2 * np.pi)
        return val

    def minimize(self):
        def energy_function(x):
            self.sigma = x[0]
            self.alpha = x[1]
            self.l = x[2]
            return self.negative_log_likelihood

        x = np.array([self.sigma, self.alpha, self.l])
        res = opt.minimize(energy_function, x)
        # trigger the function again to surely store the minimizer
        energy_function(res.x)

    def predict(self, xs):
        sigma = self.sigma
        alpha = self.alpha
        l = self.l
        beta_inv = self.beta_inv

        xb = np.tile(xs, self.x.shape).reshape(self.x.shape + (-1, ))
        xa = np.tile(self.x, xs.shape).reshape(xs.shape + (-1, )).T
        kern = self.kernel_function(xa, xb, sigma, alpha, l)
        mean = kern.T @ self.inv_covariance_matrix @ self.y

        xb_star = np.tile(xs, xs.shape).reshape(xs.shape + (-1, ))
        xa_star = xb_star.T
        kern_star = self.kernel_function(xa_star, xb_star, sigma, alpha, l)
        var = kern_star + beta_inv - kern.T @ self.inv_covariance_matrix @ kern
        std = np.sqrt(np.diag(var))

        return mean, std


# ------------------------------- Visualization ------------------------------ #


def visualize(x, y, xs, ymean, ystd, title=None):
    plt.figure(title)
    plt.title(title)
    plt.plot(xs, ymean)
    plt.fill_between(xs,
                     ymean + 2 * ystd,
                     ymean - 2 * ystd,
                     facecolor='lightsalmon')
    plt.ylim(-4, 4)
    plt.xlim(-60, 60)
    plt.scatter(x, y, color='magenta')


# ------------------------------- Main Function ------------------------------ #

if __name__ == '__main__':
    x, y = load_data(INPUT_FILENAME)

    gp = GaussianProcess(x, y, BETA, SIGMA_INIT, ALPHA_INIT, L_INIT)

    xs = np.linspace(-60, 60, 1000)
    ymean, ystd = gp.predict(xs)
    visualize(x, y, xs, ymean, ystd, title='Initial guess')

    # minimize parameter sigma, alpha, and l
    gp.minimize()

    ymean, ystd = gp.predict(xs)
    visualize(x, y, xs, ymean, ystd, title='Optimized')
    plt.show()
