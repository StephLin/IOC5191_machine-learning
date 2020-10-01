# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 01
# Student ID: 309553002
# ---------------------------------------------------------------------------- #

import argparse
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

Data = namedtuple('Data', ['points', 'n', 'regularizer'])

# ----------------------------- Utility functions ---------------------------- #


def get_points(filename: str) -> np.ndarray:
    points = []
    with open(filename, 'r') as f:
        for line in f:
            points.append(list(map(float, line.split(','))))
    return np.array(points)


def plot_message(data: Data, solution: np.ndarray):
    sol = solution.T[0]
    display_list = [f'{c:.5f}X^{d}' for c, d in zip(sol, range(data.n))]
    display_list = display_list[::-1]
    display_message = (' + '.join(display_list))[:-3]  # remove X^0
    print('Fitting line:', display_message)

    error = total_error(data, solution)
    print(f'Total error: {error:.5f}')


# ----------------------------- Matrix operators ----------------------------- #


def is_matrix(array: np.ndarray):
    return len(array.shape) == 2


def matrix_multiply(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    assert is_matrix(mat_a)
    assert is_matrix(mat_b)
    assert mat_a.shape[1] == mat_b.shape[0]

    mat = np.zeros((mat_a.shape[0], mat_b.shape[1]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j] = np.sum(mat_a[i, :] * mat_b[:, j])
    return mat


def minor(mat: np.ndarray, i: int, j: int) -> np.ndarray:
    assert is_matrix(mat)

    rows = np.concatenate([mat[:i], mat[i + 1:]])
    matrix = [np.concatenate([row[:j], row[j + 1:]]) for row in rows]
    return np.array(matrix)


def det(mat: np.ndarray) -> np.ndarray:
    assert is_matrix(mat)
    assert mat.shape[0] == mat.shape[1]

    if mat.shape[0] == 1:
        return mat[0, 0]
    value = 0
    for j, element in enumerate(mat[0]):
        theta = -2 * (j % 2) + 1
        value += theta * element * det(minor(mat, 0, j))
    return value


def adjoint_matrix(mat: np.ndarray) -> np.ndarray:
    assert is_matrix(mat)
    assert mat.shape[0] == mat.shape[1]

    adj = np.zeros(mat.shape)
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            adj[i, j] = (-1)**(i + j) * det(minor(mat, i, j))

    return adj


def matrix_inverse(mat: np.ndarray) -> np.ndarray:
    return adjoint_matrix(mat) / det(mat)


# ----------------------- Linear System and Total Error ---------------------- #


def build_matrix(data: Data) -> np.ndarray:
    matrix = [data.points[:, 0]**i for i in range(data.n)]
    return np.array(matrix).T


def build_vector(data: Data) -> np.ndarray:
    return data.points[:, 1, np.newaxis]


def total_error(data: Data, coeffs: np.ndarray) -> np.ndarray:
    @np.vectorize
    def polynomial(x):
        return np.sum([x**i * coeff for i, coeff in enumerate(coeffs)])

    lse = polynomial(data.points[:, 0]) - data.points[:, 1]
    return np.sum(lse**2)


# --------------------- Least Squares and Newton's Method -------------------- #


def least_squares(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    normal_matrix = matrix_multiply(matrix.T, matrix)
    normal_matrix += data.regularizer * np.eye(data.n)
    normal_vector = matrix_multiply(matrix.T, vector)

    solution = matrix_multiply(matrix_inverse(normal_matrix), normal_vector)
    return solution


def newton_method(matrix: np.ndarray,
                  vector: np.ndarray,
                  x0: np.ndarray,
                  epsilon: float = 1e-8) -> np.ndarray:
    x = x0
    gap_norm = epsilon * 2
    while gap_norm > epsilon:
        hessian = 2 * matrix_multiply(matrix.T, matrix)
        gradient = matrix_multiply(hessian, x)
        gradient -= 2 * matrix_multiply(matrix.T, vector)

        gap = matrix_multiply(matrix_inverse(hessian), gradient)
        gap_norm = np.sqrt(np.sum(gap**2))
        x -= gap
    return x


# ------------------------------- Visualization ------------------------------ #


def plot_figure(data: Data,
                solution: np.ndarray,
                title: str = '',
                margin: int = 1,
                density: int = 500):
    plt.figure(title)
    plt.title(title)
    fitting = np.poly1d(solution.T[0][::-1])
    xmin = np.min(data.points[:, 0]) - margin
    xmax = np.max(data.points[:, 0]) + margin
    xs = np.linspace(xmin, xmax, density)

    plt.scatter(data.points[:, 0], data.points[:, 1], color='C1')
    plt.plot(xs, fitting(xs), color='C0')
    # plt.savefig(f'{title}.png', dpi=500)


# ------------------------------- Main function ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='file which consists of data points')
    parser.add_argument('n', help='degree', type=int)
    parser.add_argument('regularizer',
                        help='regularizer (only for LSE)',
                        type=int)

    args = parser.parse_args()
    points = get_points(args.filename)
    data = Data(points=points, n=args.n, regularizer=args.regularizer)

    # build linear system
    matrix = build_matrix(data)
    vector = build_vector(data)

    print('LSE:')
    solution = least_squares(matrix, vector)
    plot_message(data, solution)
    plot_figure(data, solution, title='LSE')
    print()

    print('Newton\'s Method:')
    solution = newton_method(matrix, vector, np.ones((data.n, 1)))
    plot_message(data, solution)
    plot_figure(data, solution, title='Newton\'s Method')

    plt.show()
