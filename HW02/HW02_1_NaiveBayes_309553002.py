# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 02-1: Naive Bayes Classifier
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import os.path
import pickle
import gzip
import logging
from typing import Any, List
from collections import namedtuple
import argparse

import numpy as np

MNIST_TRAIN_IMAGE_FILENAME = './train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABEL_FILENAME = './train-labels-idx1-ubyte.gz'
MNIST_T10K_IMAGE_FILENAME = './t10k-images-idx3-ubyte.gz'
MNIST_T10K_LABEL_FILENAME = './t10k-labels-idx1-ubyte.gz'

MNIST_TRAIN_CACHE_FILENAME = './train.pickle'
MNIST_T10K_CACHE_FILENAME = './t10k.pickle'

DISCRETE_MODE = 'discrete'
CONTINUOUS_MODE = 'continuous'
PREPROCESSING_MODE = DISCRETE_MODE  # options: `DISCRETE_MODE`, `CONTINUOUS_MODE`

BINARY_IMAGE_THRESHOLD = 128

CONTINUOUS_VARIANCE_TOLERANCE = 2500
DISCRETE_VARIANCE_TOLERANCE = 36

LOGGING_LEVEL = logging.INFO

logger = logging.getLogger('NaiveBayes')
mnist = namedtuple('mnist', ['images', 'labels'])

# ----------------------------- Utility functions ---------------------------- #


def calc_mean(data: Any):
    return np.sum(data) / np.prod(data.shape)


def calc_variance(data: Any, tol: float = 1e-5):
    mu = calc_mean(data)
    var = np.sum((data - mu)**2) / np.prod(data.shape)
    return np.max([var, tol])


def gaussian(x, mean: Any, variance: Any, tol: float = 1e-10):
    scale = 1 / np.sqrt(2 * np.pi * variance)
    val = scale * np.exp(-0.5 * (x - mean)**2 / variance)
    return np.maximum(val, tol)


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


# -------------------------------- Naive Bayes ------------------------------- #


class NaiveBayesClassifier(object):
    def __init__(self, dataset: mnist, variance_tolerance: float):
        self.dataset = dataset
        self.number_of_images = dataset.images.shape[0]
        self.labels = np.array(list(set(dataset.labels)))
        self.number_of_labels = self.labels.shape[0]
        self.variance_tolerance = variance_tolerance

        pdf_shape = dataset.images.shape[1:] + (self.number_of_labels, 256)
        self.stats = np.zeros(pdf_shape)
        self.pdf = np.zeros(pdf_shape[:2] + (10, 2))

    def train(self):
        logger.debug("training naive bayes")
        train_images = self.dataset.images
        train_labels = self.dataset.labels

        for row in range(train_images.shape[1]):
            for column in range(train_images.shape[2]):
                for i, label in enumerate(self.labels):
                    indices = np.where(train_labels == label)[0]
                    data = train_images[indices, row, column].flatten()
                    mean = calc_mean(data)
                    variance = calc_variance(data, self.variance_tolerance)
                    self.pdf[row, column, i] = [mean, variance]

    def predict(self, dataset: mnist) -> List[np.ndarray]:
        logger.debug("predicting via naive bayes")
        number_of_images = dataset.images.shape[0]
        result = np.zeros([number_of_images, self.number_of_labels])
        indices = np.indices(dataset.images.shape[1:])
        for idx, image in enumerate(dataset.images):
            means = self.pdf[indices[0], indices[1], :, 0]
            variances = self.pdf[indices[0], indices[1], :, 1]
            images = np.zeros(means.shape)
            for i in range(self.number_of_labels):
                images[:, :, i] = image
            prob = gaussian(images, means, variances)
            log_prob = np.log(prob).sum(axis=0).sum(axis=0)
            result[idx] = log_prob / np.sum(log_prob)
        return result, self.labels[result.argmin(axis=1)]


# ------------------------------- Visualization ------------------------------ #


def plot_result(posteriori, predict, answer, label_names):
    print('Posteriori (in log scale):')
    for name, value in zip(label_names, posteriori):
        print('%s: %.5f' % (name, value))
    print('Prediction: %s, Ans: %s\n' % (predict, answer))


def plot_binary_images(predict: np.ndarray, dataset: mnist,
                       label_names: np.ndarray, binary_image_threshold: int):

    print('Imagination of numbers in Bayesian classifier:\n')
    for name in label_names:
        indices = np.where(predict == name)[0]
        idx = None
        for idx in indices:
            if predict[idx] == dataset.labels[idx]:
                break
        print('%s:' % name)
        image = np.where(dataset.images[idx] >= binary_image_threshold, 1, 0)
        for row in image:
            print(' '.join(map(str, row)))
        print()


# ------------------------------- Main function ------------------------------ #

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        type=str,
        nargs='?',
        const=1,
        default=PREPROCESSING_MODE,
        help=('mode for image preprocessing '
              '(`%s` or `%s`), default as `%s`.' %
              (DISCRETE_MODE, CONTINUOUS_MODE, PREPROCESSING_MODE)))
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

    logger.debug("use %s mode for image preprocessing" % args.mode)

    # load training and testing sets
    train = load_mnist_dataset_via_cache_or_file(MNIST_TRAIN_IMAGE_FILENAME,
                                                 MNIST_TRAIN_LABEL_FILENAME,
                                                 MNIST_TRAIN_CACHE_FILENAME,
                                                 args.nocache)
    t10k = load_mnist_dataset_via_cache_or_file(MNIST_T10K_IMAGE_FILENAME,
                                                MNIST_T10K_LABEL_FILENAME,
                                                MNIST_T10K_CACHE_FILENAME,
                                                args.nocache)

    # handling continuous and discrete mode
    if args.mode == CONTINUOUS_MODE:
        variance_tolerance = CONTINUOUS_VARIANCE_TOLERANCE
        binary_image_threshold = BINARY_IMAGE_THRESHOLD
    elif args.mode == DISCRETE_MODE:
        train.images[:] = train.images // 8
        t10k.images[:] = t10k.images // 8
        variance_tolerance = DISCRETE_VARIANCE_TOLERANCE
        binary_image_threshold = BINARY_IMAGE_THRESHOLD // 8
    else:
        raise ValueError("invalid mode")

    # train and predict via Naive Bayes classifier
    classifier = NaiveBayesClassifier(train, variance_tolerance)
    classifier.train()
    result = classifier.predict(t10k)

    # plot log scale posteriori, prediction, and answer
    for posteriori, predict, answer in zip(result[0], result[1], t10k.labels):
        plot_result(posteriori, predict, answer, classifier.labels)

    # plot imagination of numbers in classifier
    plot_binary_images(result[1], t10k, classifier.labels,
                       binary_image_threshold)

    # calculate precision and plot error rate
    precision = np.sum(result[1] == t10k.labels) / t10k.labels.shape[0]
    error_rate = 1 - precision
    print('Error rate: %.5f' % error_rate)
