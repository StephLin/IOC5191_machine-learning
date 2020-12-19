# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 06: Spectral Clustering
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import os.path as osp
import logging
import argparse
import numpy as np
from scipy.spatial.distance import cdist
import imageio
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('AGG')

logging.basicConfig()

logger = logging.getLogger('SpectralClustering')

# ---------------------------- Data Preprocessing ---------------------------- #


def load_image(filename):
    img = imageio.imread(filename)
    return img[..., :3]


def add_spatial_channels(img):
    super_image = np.zeros(img.shape[:2] + (5, ))
    super_image[..., :3] = img[..., :3]

    indices = np.indices(img.shape[:2])
    super_image[..., 3] = indices[0]
    super_image[..., 4] = indices[1]
    return super_image


def load_data(filename):
    img = load_image(filename)
    data = add_spatial_channels(img)
    return data


# ---------------------------------- Kernel ---------------------------------- #


def gram_matrix(data, gamma_c=1e-3, gamma_s=1e-3):
    size = data.shape[0]
    matrix = np.zeros((size, size))
    color_distance = cdist(data[:, :3], data[:, :3], metric='sqeuclidean')
    spatial_distance = cdist(data[:, 3:], data[:, 3:], metric='sqeuclidean')
    color_kernel = np.exp(-gamma_c * color_distance)
    spatial_distance = np.exp(-gamma_s * spatial_distance)
    matrix = color_kernel * spatial_distance
    return matrix


# ---------------------------------- K-Means --------------------------------- #


class KMeans(object):
    def __init__(self, data, k, init='random', tol=1e-3):
        self.data = data
        self.size = data.shape[0]
        self.degree = data.shape[1]
        self.k = k
        self.cluster_means = self._get_init_cluster_means(init)
        self.label = None
        self.label_history = []
        self.tol = tol

    @property
    def label_matrix(self):
        matrix = np.zeros((self.size, self.k), dtype=int)
        matrix[np.arange(self.size), self.label] = 1
        return matrix

    @property
    def label_count(self):
        return self.label_matrix.sum(axis=0)

    def _get_init_cluster_means(self, init='random'):
        cluster_means = np.zeros((self.k, self.degree))
        if init == 'random':
            logger.debug('use random initialization for k-means')
            indices = np.random.randint(0, self.size, self.k)
            cluster_means[:] = self.data[indices]
        elif init == 'k-means++':
            logger.debug('use k-means++ initialization for k-means')
            idx = np.random.randint(0, self.size, 1)[0]
            cluster_means[0] = self.data[idx]
            for i in range(1, self.k):
                distance = np.zeros((self.size, i))
                for j in range(i):
                    diff = self.data - cluster_means[j]
                    distance[:, j] = np.sum(diff**2, axis=1)
                distance = np.min(distance, axis=1)
                distance = distance / np.sum(distance)
                print(distance)
                idx = np.random.choice(np.arange(self.size), p=distance)
                # idx = np.argmax(distance)
                cluster_means[i] = self.data[idx]
        else:
            raise NotImplementedError("Unknown init type")

        return cluster_means

    def _expectation_step(self):
        distance = np.zeros((self.size, self.k))
        for i in range(self.k):
            diff = self.data - self.cluster_means[i]
            distance[:, i] = np.linalg.norm(diff, axis=1)
        label = np.argmin(distance, axis=1)
        return label

    def _maximiation_step(self, label):
        cluster_means = np.zeros((self.k, self.degree))
        for i in range(self.k):
            indices = np.argwhere(label == i).flatten()
            cluster_means[i] = np.mean(self.data[indices], axis=0)
        return cluster_means

    def train(self, max_iter=100):
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            logger.debug('train iter [%d/%d]' % (n_iter, max_iter))
            label = self._expectation_step()
            cluster_means = self._maximiation_step(label)

            # handling first iteration
            if self.label is None:
                self.label = label
                self.cluster_means = cluster_means
                self.label_history.append(label)
                continue

            # check if converge
            err = len(np.where(self.label != label)[0]) / self.size
            logger.debug('err: %f' % err)
            if err < self.tol:
                logger.debug('training terminated due to convergence')
                self.label = label
                self.cluster_means = cluster_means
                self.label_history.append(label)
                break

            self.label = label
            self.cluster_means = cluster_means
            self.label_history.append(label)

    def visualize_as_gif(self, image, filename):
        data = image.reshape((-1, 3))
        label_color = (self.label_matrix.T @ data)[:, :3]
        label_color /= self.label_count[:, np.newaxis]
        label_color = label_color.astype(np.uint8)
        with imageio.get_writer(filename, mode='I') as writer:
            for label in self.label_history:
                image_ = label_color[label.reshape(image.shape[:2])]
                writer.append_data(image_)

    def visualize_as_image(self, image, filename):
        data = image.reshape((-1, 3))
        label_color = (self.label_matrix.T @ data)[:, :3]
        label_color /= self.label_count[:, np.newaxis]
        label_color = label_color.astype(np.uint8)

        image_ = label_color[self.label.reshape(image.shape[:2])]
        imageio.imwrite(filename, image_)


# ---------------------------- Spectral Clustering --------------------------- #


class SpectralClustering(object):
    def __init__(self,
                 gram,
                 k,
                 kmeans_init='random',
                 mode='unnormalized',
                 tol=1e-3):
        self.gram = gram
        self.size = gram.shape[0]
        self.k = k
        self.kmeans_init = kmeans_init
        self.tol = tol
        self.degree_matrix = np.diag(gram.sum(axis=1))
        self.graph_laplacian = self._calc_graph_laplacian(mode)
        self.features = np.zeros((self.size, self.k))

        self.eigvals = None
        self.eigvecs = None

        self.km = None

    @property
    def label(self):
        return self.km.label

    @property
    def label_history(self):
        return self.km.label_history

    def _calc_graph_laplacian(self, mode='unnormalized'):
        graph_laplacian = self.degree_matrix - self.gram
        if mode == 'unnormalized':
            pass
        elif mode == 'normalized':
            d = np.diag(1 / np.sqrt(np.diag(self.degree_matrix)))
            graph_laplacian = d @ graph_laplacian @ d
        else:
            raise NotImplementedError("Unknown clustering mode")

        return graph_laplacian

    def train(self):
        # solve the eigenvalue problem
        if self.eigvals is None or self.eigvecs is None:
            eigvals, eigvecs = np.linalg.eig(self.graph_laplacian)
            self.eigvals = eigvals
            self.eigvecs = eigvecs

        # pick up the first k nonzero eigenvalues with ones corresponding
        # eigenvectors, and generate the feature mapping
        indices = np.argsort(self.eigvals)
        self.eigvals = self.eigvals[indices]
        self.eigvecs = self.eigvecs[:, indices].real
        base_idx = np.where(self.eigvals > 1e-8)[0][0]
        self.features = self.eigvecs[:, base_idx:base_idx + self.k]

        # trigger the K-Means algorithm
        self.km = KMeans(self.features, self.k, self.kmeans_init, self.tol)
        self.km.train()

    def visualize_as_gif(self, image, filename):
        self.km.visualize_as_gif(image, filename)

    def visualize_as_image(self, image, filename):
        self.km.visualize_as_image(image, filename)

    def visualize_eigenspace_as_image(self, filename):
        plt.figure()
        plt.xlabel('fisrt non-null eigenvector')
        plt.ylabel('second non-null eigenvector')
        coord = self.features[:, :2]
        for i in range(self.k):
            indices = np.argwhere(self.label == i).flatten()
            plt.scatter(coord[indices, 0], coord[indices, 1])
        plt.savefig(filename, dpi=500)


# ------------------------------- Main Function ------------------------------ #

if __name__ == '__main__':
    # fixed random seed for debugging
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='path to image file')
    parser.add_argument('k', type=int, help='number of clusters')
    parser.add_argument('mode', type=str, help='clustering mode')
    parser.add_argument('init', type=str, help='methods of initialization')
    parser.add_argument('--debug',
                        action='store_true',
                        help='show debug message')
    parser.add_argument('--cache',
                        action='store_true',
                        help='use cached eigenvalues and eigenvectors')
    parser.add_argument('--save-cache',
                        action='store_true',
                        help='use cached eigenvalues and eigenvectors')

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    filename = args.filename
    k = args.k
    mode = args.mode
    init = args.init

    raw_filename = osp.splitext(filename)[0]
    gif_filename = 'images/%s_%d_%s_%s_sc_result.gif' % (raw_filename, k, init,
                                                         mode)
    png_filename = 'images/%s_%d_%s_%s_sc_result.png' % (raw_filename, k, init,
                                                         mode)
    eig_filename = 'images/%s_%d_%s_%s_sc_eig_result.png' % (raw_filename, k,
                                                             init, mode)
    cache_eigvals_filename = '%s.%s.eigvals.npy' % (raw_filename, mode)
    cache_eigvecs_filename = '%s.%s.eigvecs.npy' % (raw_filename, mode)

    data = load_data(filename)
    logger.debug('calculating gram matrix')
    gram = gram_matrix(data.reshape(-1, 5))

    logger.debug('initializing spectial clustering')
    sc = SpectralClustering(gram, k, init, mode)

    if args.cache:
        sc.eigvals = np.load(cache_eigvals_filename)
        sc.eigvecs = np.load(cache_eigvecs_filename)

    logger.debug('trigger clustering')
    sc.train()

    sc.visualize_as_gif(data[..., :3], gif_filename)
    sc.visualize_as_image(data[..., :3], png_filename)
    sc.visualize_eigenspace_as_image(eig_filename)

    if args.save_cache:
        np.save(cache_eigvals_filename, sc.eigvals)
        np.save(cache_eigvecs_filename, sc.eigvecs)
