# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 04-2: EM Algorithm
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
from os import major
import os.path
import pickle
import gzip
import logging
from collections import namedtuple
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

MNIST_TRAIN_IMAGE_FILENAME = './train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABEL_FILENAME = './train-labels-idx1-ubyte.gz'

MNIST_TRAIN_CACHE_FILENAME = './train.pickle'

BINARY_IMAGE_THRESHOLD = 128

LOGGING_LEVEL = logging.INFO

logger = logging.getLogger('NaiveBayes')
mnist = namedtuple('mnist', ['images', 'labels'])

# ----------------------------- Utility functions ---------------------------- #


def gaussian(x: np.ndarray, mean: np.ndarray,
             variance: np.ndarray) -> np.ndarray:
    scale = 1 / np.sqrt(2 * np.pi * np.linalg.det(variance))
    lambda_ = np.linalg.inv(variance)
    val = scale * np.exp(-0.5 * (x - mean) @ lambda_ @ (x - mean))
    return val


# ----------------------------------- MNIST ---------------------------------- #


def _load_mnist_dataset(image_filename: str, label_filename: str) -> mnist:
    images = None
    labels = None

    logger.debug("reading image data")
    with gzip.open(image_filename, 'rb') as f:
        f.read(4)  # magic number
        number_of_images = int.from_bytes(f.read(4), 'big')
        number_of_rows = int.from_bytes(f.read(4), 'big')
        number_of_columns = int.from_bytes(f.read(4), 'big')

        images = np.zeros(
            (number_of_images, number_of_rows, number_of_columns),
            dtype=np.uint8)
        for i in range(number_of_images):
            for row in range(number_of_rows):
                for column in range(number_of_columns):
                    images[i, row, column] = int.from_bytes(f.read(1), 'big')

    logger.debug("reading label data")
    with gzip.open(label_filename, 'rb') as f:
        f.read(4)  # magic number
        number_of_items = int.from_bytes(f.read(4), 'big')

        assert number_of_images == number_of_items

        labels = np.zeros(number_of_items, dtype=np.uint8)
        for i in range(number_of_items):
            labels[i] = int.from_bytes(f.read(1), 'big')

    return mnist(images=images, labels=labels)


def load_mnist_dataset_via_cache_or_file(image_filename: str,
                                         label_filename: str,
                                         cache_filename: str,
                                         nocache: bool = False) -> mnist:
    dataset = mnist(images=None, labels=None)

    if os.path.isfile(cache_filename) and not nocache:
        logger.debug('use cache file %s for dataset' % cache_filename)
        dataset = pickle.load(open(cache_filename, 'rb'))
    else:
        logger.debug('read gz files %s, %s for training dataset' %
                     (image_filename, label_filename))
        dataset = _load_mnist_dataset(image_filename, label_filename)
        if not nocache:
            pickle.dump(dataset, open(cache_filename, 'wb'))
    return dataset


# ------------------------------- Visualization ------------------------------ #


def plot_binary_images(image):
    for row in image:
        print(' '.join(map(str, row)))
    print()


def plot_predict_result(dataset: mnist, label_idx: int, indices: np.ndarray):
    print('%s %d:' % ('labeled class', label_idx))
    image = np.zeros(dataset.images[0].shape)
    for idx in indices:
        if dataset.labels[idx] == label_idx:
            image = dataset.images[idx]
            break
    plot_binary_images(image)


# ------------------------------- EM Algorithm ------------------------------- #


class EMClassifier(object):
    def __init__(self,
                 dataset: mnist,
                 n_classes: int,
                 tol: float = 1e-20,
                 convergence_rate: float = 0.02):
        self.dataset = dataset
        self.data = dataset.images.reshape((dataset.images.shape[0], -1))
        self.n_classes = n_classes
        self.tol = tol
        self.exp_tol = 1e-323
        self.convergence_rate = convergence_rate
        dsize = self.data.shape[0]
        degree = self.data.shape[-1]
        self.dsize = dsize

        self.p = np.random.uniform(tol, 1, size=(n_classes, degree))
        self.ld = np.ones(n_classes) / n_classes  # lambda
        self.w = np.random.uniform(tol, 1, size=(n_classes, dsize))

    def train(self, max_iter: int = 30, verbose: bool = False):
        w = self.w
        ld = self.ld
        p = self.p
        w_diff = 0
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            w = self.expectation_step()
            ld, p = self.maximization_step(w)

            w_diff = self.w_diff(w)
            if self.is_converge(w):
                break

            self.w = w
            self.ld = ld
            self.p = p
            self.iter_report(n_iter, w_diff)

        self.w = w
        self.ld = ld
        self.p = p
        self.iter_report(n_iter, w_diff)
        return n_iter

    def expectation_step(self):
        w = np.zeros(self.w.shape)
        for k in range(self.n_classes):
            w_ = self.data * self.p[k] + (1 - self.data) * (1 - self.p[k])
            w_ = np.where(w_ > self.exp_tol, w_, self.exp_tol)
            w_ = np.log(w_)
            w[k] = w_.sum(axis=1)
            w[k] += np.log(self.ld[k])
        exp_max = w.max(axis=0)
        w -= exp_max
        w = np.exp(w)
        w /= w.sum(axis=0)
        return w

    def maximization_step(self, w=None):
        if w is None:
            w = self.w

        ld = w.sum(axis=1) / self.data.shape[0]
        p = np.zeros(self.p.shape)
        for k in range(self.n_classes):
            divider = w[k].sum()
            divider = np.where(divider > 1e-30, divider, 1)
            p[k] = (self.data.T * w[k]).sum(axis=1) / divider
        return ld, p

    def w_diff(self, w):
        return np.sum(np.abs(self.w - w))

    def is_converge(self, w):
        w_diff = self.w_diff(w)
        return w_diff < self.convergence_rate * self.data.shape[0]

    @property
    def raw_classes(self):
        return np.argmax(self.w, axis=0)

    @property
    def major_label(self):
        raw_classes = self.raw_classes

        # Hungarian algorithm
        cost_matrix = np.ones((self.n_classes, self.n_classes)) * 100
        for i in range(self.n_classes):
            indices = np.argwhere(raw_classes == i).flatten()
            indices = self.dataset.labels[indices]
            indeed, count = np.unique(indices, return_counts=True)
            cost_matrix[i, indeed] = 1 / count
        _, major_label = linear_sum_assignment(cost_matrix)

        return major_label

    @property
    def labeled_classes(self):
        return self.major_label[self.raw_classes]

    @property
    def error_rate(self):
        labels = self.labeled_classes
        error = np.where(labels == self.dataset.labels, 0, 1)
        return error.sum() / self.dsize

    def iter_report(self, n_iter: int, w_diff: float):
        labels = self.labeled_classes
        for i in range(self.n_classes):
            indices = np.argwhere(labels == i).flatten()
            plot_predict_result(self.dataset, i, indices)

        print('No. of Iteration: %d' % n_iter, end=', ')
        print('w 1-Norm Difference: %f' % w_diff, end=', ')
        print('Total Error Rate: %f' % self.error_rate)
        print('\n%s\n' % ('-' * 60))

    def summary(self):
        predict = self.labeled_classes
        truth = self.dataset.labels
        for i in range(self.n_classes):
            tp = np.where((predict == i) & (truth == i), 1, 0).sum()
            fn = np.where((predict != i) & (truth == i), 1, 0).sum()
            fp = np.where((predict == i) & (truth != i), 1, 0).sum()
            tn = np.where((predict != i) & (truth != i), 1, 0).sum()

            print('\nConfusion Matrix %d:' % i)
            print('%15s  %16s  %20s' % (' ' * 15, 'Predict number %d' % i,
                                        'Predict not number %d' % i))
            print('%15s  %16d  %20d' % ('Is number %d' % i, tp, fn))
            print('%15s  %16d  %20d\n' % ('Isn\'t number %d' % i, fp, tn))

            sens = tp / (tp + fn)
            spec = tn / (fp + tn)
            print('Sensitivity (Successfully predict number %d)    : %5f' %
                  (i, sens))
            print('Specificity (Successfully predict not number %d): %5f' %
                  (i, spec))
            print('\n%s\n' % ('-' * 60))


# ------------------------------- Main function ------------------------------ #

if __name__ == '__main__':

    # ensure that each run is predictable
    np.random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nocache',
                        dest='nocache',
                        action='store_true',
                        default=False,
                        help='do not use pickle as cache feature')
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='show debugging message')
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else LOGGING_LEVEL
    logging.basicConfig(level=logging_level)

    # load training and testing sets
    train = load_mnist_dataset_via_cache_or_file(MNIST_TRAIN_IMAGE_FILENAME,
                                                 MNIST_TRAIN_LABEL_FILENAME,
                                                 MNIST_TRAIN_CACHE_FILENAME,
                                                 args.nocache)
    train.images[:] = np.where(train.images > BINARY_IMAGE_THRESHOLD, 1, 0)
    train.images[:] = train.images.astype(int)

    em = EMClassifier(train, 10)
    n_iter = em.train()
    em.summary()
    print('Total iteration to converge: %d' % n_iter)
    print('Total error rate: %5f' % em.error_rate)
