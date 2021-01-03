# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 07-1: Kernel Eigenfaces
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import glob
import logging
import os.path as osp
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

logging.basicConfig()
logger = logging.getLogger('KernelEigenFaces')

SHAPE = (50, 50)

yale = namedtuple('yale', ['data', 'label', 'image_shape'])

# ---------------------------- Yale Face Database ---------------------------- #


def _load_data(directory, number_of_subjects):
    data = []
    label = []
    for idx in range(1, number_of_subjects + 1):
        pattern = 'subject%02d*.pgm' % idx
        path = osp.join(directory, pattern)
        filenames = glob.glob(path)
        filenames.sort()
        for filename in filenames:
            label.append(idx)
            image = Image.open(filename)
            image = np.array(image.resize(SHAPE, Image.ANTIALIAS))
            image = image.astype(float) / 255
            data.append(image)

    data = np.array(data)
    label = np.array(label)

    image_shape = data.shape[1:]
    data = data.reshape((data.shape[0], -1))

    dataset = yale(data, label, image_shape)

    return dataset


def load_data(database_root='./Yale_Face_Database',
              training_dir='Training',
              testing_dir='Testing',
              number_of_subjects=15):
    training_dataset = _load_data(osp.join(database_root, training_dir),
                                  number_of_subjects)
    testing_dataset = _load_data(osp.join(database_root, testing_dir),
                                 number_of_subjects)

    return training_dataset, testing_dataset


# ----------------------------------- K-NN ----------------------------------- #


def knn(source, label, targets, k=7):
    result = []
    for target in targets:
        distance = np.linalg.norm(source - target, axis=1)
        indices = np.argsort(distance)[:k]
        labels = label[indices]
        voting = np.argmax(np.bincount(labels))
        result.append(voting)

    return np.array(result)


# -------------------------------- PCA and LDA ------------------------------- #


class PCA(object):
    def __init__(self, data, dimension=25):
        self.data = data
        self.mean = data.mean(axis=0)
        self.dimension = dimension
        self.singular_values = np.array([])
        self.left_singular_vectors = np.array([])
        self.right_singular_vectors = np.array([])

    @property
    def feature_vectors(self):
        return self.right_singular_vectors[:, :self.dimension]

    @property
    def feature_coordinates(self):
        return self.dimensionality_reduction(self.data)

    def dimensionality_reduction(self, data):
        return (data - self.mean) @ self.feature_vectors

    def compute(self):
        matrix = self.data - self.mean
        cov = matrix @ matrix.T

        eigvals, eigvecs = np.linalg.eig(cov)
        eigvals = np.where(eigvals < 0, 0, eigvals)

        indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        self.singular_values = np.sqrt(eigvals)
        self.left_singular_vectors = eigvecs
        self.right_singular_vectors = self.data.T @ eigvecs

        for i in range(self.right_singular_vectors.shape[1]):
            vector = self.right_singular_vectors[:, i]
            vector = vector / np.linalg.norm(vector)
            self.right_singular_vectors[:, i] = vector


class LDA(object):
    def __init__(self, data, label, dimension=25):
        self.data = data
        self.mean = data.mean(axis=0)
        self.degree = data.shape[1]
        self.label = label
        self.dimension = dimension

        self.sw = np.array([])
        self.sb = np.array([])

        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([])

    @property
    def feature_vectors(self):
        return self.eigenvectors[:, :self.dimension]

    @property
    def feature_coordinates(self):
        return self.dimensionality_reduction(self.data)

    def dimensionality_reduction(self, data):
        return (data - self.mean) @ self.feature_vectors

    def compute(self):
        self.sw = np.zeros((self.degree, self.degree), dtype=float)
        self.sb = np.zeros((self.degree, self.degree), dtype=float)

        unique_labels = np.unique(self.label)
        for label in unique_labels:
            indices = np.argwhere(self.label == label).flatten()
            data = self.data[indices]
            mu = np.mean(data, axis=0)
            self.sw += (data - mu).T @ (data - mu)
            diff = np.array([mu - self.mean]).T
            self.sb += len(indices) * diff @ diff.T
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(self.sw) @ self.sb)

        # normalize eigenvectors
        eigvecs = eigvecs.real
        for idx in range(eigvecs.shape[1]):
            eigvecs[:, idx] /= np.linalg.norm(eigvecs[:, idx])

        # sort eigenvalues and eigenvectors
        indices = np.argsort(eigvals)[::-1]

        self.eigenvalues = eigvals[indices]
        self.eigenvectors = eigvecs[:, indices]


# ------------------------------- Kernel tricks ------------------------------ #


def linear_kernel(x, y=None):
    y = x if y is None else y
    return cdist(x, y, lambda u, v: u @ v)


def polynomial_kernel(x, y=None, c=1, d=1):
    y = x if y is None else y
    return cdist(x, y, lambda u, v: (u @ v + c)**d)


def rbf_kernel(x, y=None, gamma=1e-3):
    y = x if y is None else y
    dist = cdist(x, y, 'sqeuclidean')
    return np.exp(-gamma * dist)


class Kernel(object):
    LINEAR = linear_kernel
    POLYNOMIAL = polynomial_kernel
    RBF = rbf_kernel


class KernelPCA(object):
    def __init__(self, data, kernel_type, dimension=25):
        self.data = data
        self.dimension = dimension
        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([])

        self.kernel_function = kernel_type
        self.kernel = self.kernel_function(self.data)
        self.centralized_kernel = self.centralize(self.kernel)

    def centralize(self, kernel):
        # https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        ones = np.ones(kernel.shape) / kernel.shape[0]
        return kernel - ones @ kernel - kernel @ ones + ones @ kernel @ ones

    @property
    def feature_vectors(self):
        return self.eigenvectors[:, :self.dimension]

    @property
    def feature_coordinates(self):
        return self.centralized_kernel @ self.feature_vectors

    def compute(self):
        eigvals, eigvecs = np.linalg.eig(self.centralized_kernel)
        eigvals = np.where(eigvals < 0, 0, eigvals)

        indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs.real

        for i in range(self.eigenvectors.shape[1]):
            vector = self.eigenvectors[:, i]
            vector = vector / np.linalg.norm(vector)
            self.eigenvectors[:, i] = vector


class KernelLDA(object):
    def __init__(self, data, label, kernel_type, dimension=25):
        self.data = data
        self.size = data.shape[0]
        self.degree = data.shape[1]
        self.label = label
        self.dimension = dimension

        self.m = np.array([])
        self.n = np.array([])

        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([])

        self.kernel_function = kernel_type
        self.kernel = self.kernel_function(self.data)
        self.centralized_kernel = self.centralize(self.kernel)

    def centralize(self, kernel):
        # https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        ones = np.ones(kernel.shape) / kernel.shape[0]
        return kernel - ones @ kernel - kernel @ ones + ones @ kernel @ ones

    @property
    def feature_vectors(self):
        return self.eigenvectors[:, :self.dimension]

    @property
    def feature_coordinates(self):
        return self.centralized_kernel @ self.feature_vectors

    def compute(self):
        self.m = np.zeros((self.size, self.size), dtype=float)
        self.n = np.zeros((self.size, self.size), dtype=float)

        unique_labels = np.unique(self.label)
        mean = np.mean(self.kernel, axis=1)
        for label in unique_labels:
            indices = np.argwhere(self.label == label).flatten()
            kernel = self.kernel[indices]
            size = len(indices)
            mean_label = np.mean(kernel, axis=0)

            mdiff = mean_label - mean
            self.m += size * np.kron(mdiff, mdiff).reshape((self.size, -1))
            ones = np.ones((size, size)) / size
            self.n += kernel.T @ (np.eye(size) - ones) @ kernel
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(self.m) @ self.n)

        # normalize eigenvectors
        eigvecs = eigvecs.real
        for idx in range(eigvecs.shape[1]):
            eigvecs[:, idx] /= np.linalg.norm(eigvecs[:, idx])

        # sort eigenvalues and eigenvectors
        indices = np.argsort(eigvals)[::-1]

        self.eigenvalues = eigvals[indices]
        self.eigenvectors = eigvecs[:, indices]


# ------------------------------- Main Function ------------------------------ #


def main(args):
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.debug("loading Yale dataset")
    training, testing = load_data()

    pca = PCA(training.data)
    lda = LDA(training.data, training.label)

    if args.enable_part_1 or args.enable_part_2:
        logger.debug("performing PCA dimensionality reduction")
        pca.compute()

        logger.debug("performing LDA dimensionality reduction")
        lda.compute()

    if args.enable_part_1:
        logger.debug("triggering part 1")
        # eigenfaces
        _, axs = plt.subplots(5, 5)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                image = pca.feature_vectors[:, idx]
                image = image.reshape(training.image_shape)
                axs[i, j].imshow(image, cmap='gray')

        # reconstruction of PCA
        _, axs = plt.subplots(2, 5)

        indices = np.random.choice(training.data.shape[0], size=10)
        feature_coordinates = pca.feature_coordinates[indices]

        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                image = pca.feature_vectors @ feature_coordinates[
                    idx] + pca.mean
                image = image.reshape(training.image_shape)
                axs[i, j].imshow(image, cmap='gray')

        # fisherfaces
        fig, axs = plt.subplots(5, 5)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                image = lda.feature_vectors[:, idx]
                image = image.reshape(training.image_shape)
                axs[i, j].imshow(image, cmap='gray')

        # reconstruction of LDA
        fig, axs = plt.subplots(2, 5)

        feature_coordinates = lda.feature_coordinates[indices]

        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                image = lda.feature_vectors @ feature_coordinates[
                    idx] + lda.mean
                image = image.reshape(training.image_shape)
                axs[i, j].imshow(image, cmap='gray')

    if args.enable_part_2:
        logger.debug("triggering part 2")
        pca_coordinates = pca.dimensionality_reduction(testing.data)
        for k in range(3, 16, 2):
            pca_result = knn(pca.feature_coordinates, training.label,
                             pca_coordinates, k)
            acc = np.where(pca_result == testing.label, 1, 0).mean() * 100
            print('PCA with k=%02d -- Accuracy: %.2f%%' % (k, acc))

        lda_coordinates = lda.dimensionality_reduction(testing.data)
        for k in range(3, 16, 2):
            lda_result = knn(lda.feature_coordinates, training.label,
                             lda_coordinates, k)
            acc = np.where(lda_result == testing.label, 1, 0).mean() * 100
            print('LDA with k=%02d -- Accuracy: %.2f%%' % (k, acc))

    if args.enable_part_3:
        logger.debug("triggering part 3")

        # concatenate training and testing data
        data = np.vstack((training.data, testing.data))
        label = np.hstack((training.label, testing.label))
        dataset = yale(data, label, training.image_shape)

        ktypes = [Kernel.LINEAR, Kernel.POLYNOMIAL, Kernel.RBF]
        knames = ['Linear', 'Polynomial', 'RBF']
        for ktype, kname in zip(ktypes, knames):
            pca = KernelPCA(dataset.data, ktype)
            pca.compute()
            pca_training = pca.feature_coordinates[:training.data.shape[0]]
            pca_testing = pca.feature_coordinates[training.data.shape[0]:]
            for k in range(3, 16, 2):
                pca_result = knn(pca_training, training.label, pca_testing, k)
                acc = np.where(pca_result == testing.label, 1, 0).mean() * 100
                print('%s Kernel PCA with k=%02d -- Accuracy: %.2f%%' %
                      (kname, k, acc))
            print('-' * 66)

            lda = KernelLDA(dataset.data, dataset.label, ktype)
            lda.compute()
            lda_training = lda.feature_coordinates[:training.data.shape[0]]
            lda_testing = lda.feature_coordinates[training.data.shape[0]:]
            for k in range(3, 16, 2):
                lda_result = knn(lda_training, training.label, lda_testing, k)
                acc = np.where(lda_result == testing.label, 1, 0).mean() * 100
                print('%s Kernel LDA with k=%02d -- Accuracy: %.2f%%' %
                      (kname, k, acc))
            print('-' * 66)

    if args.enable_part_1:
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-part-1', action='store_true')
    parser.add_argument('--enable-part-2', action='store_true')
    parser.add_argument('--enable-part-3', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    main(args)
