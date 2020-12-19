# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 06: Kernel K-Means
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import os.path as osp
import logging
import argparse
import numpy as np
from scipy.spatial.distance import cdist
import imageio

logging.basicConfig()

logger = logging.getLogger('KernelKMeans')

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


def gram_matrix(data, gamma_c=1e-4, gamma_s=1e-3):
    size = data.shape[0]
    matrix = np.zeros((size, size))

    color_distance = cdist(data[:, :3], data[:, :3], metric='sqeuclidean')
    spatial_distance = cdist(data[:, 3:], data[:, 3:], metric='sqeuclidean')

    color_kernel = np.exp(-gamma_c * color_distance)
    spatial_distance = np.exp(-gamma_s * spatial_distance)

    matrix = color_kernel * spatial_distance
    return matrix


# ------------------------------ Kernel K-Means ------------------------------ #


class KernelKMeans(object):
    def __init__(self, data, k, gram=None, init='random', tol=1e-3):
        self.data = data
        self.size = data.shape[0]
        self.k = k
        self.gram = gram_matrix(data) if gram is None else gram
        self.label = self._get_init_label(init)
        self.label_history = [self.label]
        self.tol = tol

    @property
    def label_matrix(self):
        matrix = np.zeros((self.size, self.k), dtype=int)
        matrix[np.arange(self.size), self.label] = 1
        return matrix

    @property
    def label_count(self):
        return self.label_matrix.sum(axis=0)

    def _get_init_label(self, init='random'):
        if init == 'random':
            return np.random.randint(0, self.k, self.size, dtype=int)
        elif init == 'k-means++':
            label = -1 * np.ones(self.size)
            label = label.astype(int)
            # randomly select a data point (as the center of the first cluster)
            label[np.random.randint(0, self.size, 1)] = 0

            # choose k-1 centers
            for i in range(1, self.k):
                label_matrix = np.zeros((self.size, i), dtype=int)
                label_matrix[np.arange(self.size), label] = 1
                label_matrix[np.argwhere(label == -1).flatten()] = 0

                distance = self._calc_feature_space_distance(label_matrix)
                distance = np.min(distance, axis=1)
                distance = distance / np.sum(distance)

                idx = np.random.choice(np.arange(self.size), p=distance)
                label[idx] = i

            # filling labels
            label_matrix = np.zeros((self.size, self.k), dtype=int)
            label_matrix[np.arange(self.size), label] = 1
            label_matrix[np.argwhere(label == -1).flatten()] = 0

            distance = self._calc_feature_space_distance(label_matrix)
            label = np.argmin(distance, axis=1)

            return label

        else:
            raise NotImplementedError("Unknown init type")

    def _calc_feature_space_distance(self, label_matrix=None):
        if label_matrix is None:
            k = self.k
            label_matrix = self.label_matrix
            label_count = self.label_count
        else:
            k = label_matrix.shape[1]
            label_count = label_matrix.sum(axis=0)
        assert np.all(label_count > 0)

        # calculate distance in feature space using equation (2) in
        # https://www.cs.utexas.edu/users/inderjit/public_papers/kdd_spectral_kernelkmeans.pdf
        distance = np.zeros((self.size, k))
        distance += np.diag(self.gram)[:, np.newaxis]

        distance -= 2 * (self.gram @ label_matrix) / label_count

        scale = 1 / label_count**2
        distance += np.diag(label_matrix.T @ self.gram @ label_matrix) * scale
        return distance

    def train(self, max_iter=100):
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            logger.debug('train iter [%d/%d]' % (n_iter, max_iter))
            distance = self._calc_feature_space_distance()
            label = np.argmin(distance, axis=1)
            self.label_history.append(label)

            err = len(np.where(self.label != label)[0]) / self.size
            logger.debug('err: %f' % err)
            if err < self.tol:
                logger.debug('training terminated due to convergence')
                self.label = label
                break

            self.label = label

    def visualize_as_gif(self, image_shape, filename):
        label_color = (self.label_matrix.T @ self.data)[:, :3]
        label_color /= self.label_count[:, np.newaxis]
        label_color = label_color.astype(np.uint8)
        with imageio.get_writer(filename, mode='I') as writer:
            for label in self.label_history:
                image = label_color[label.reshape(image_shape)]
                writer.append_data(image)

    def visualize_as_image(self, image_shape, filename):
        label_color = (self.label_matrix.T @ self.data)[:, :3]
        label_color /= self.label_count[:, np.newaxis]
        label_color = label_color.astype(np.uint8)

        image = label_color[self.label.reshape(image_shape)]
        imageio.imwrite(filename, image)


# ------------------------------- Main Function ------------------------------ #

if __name__ == '__main__':
    # fixed random seed for debugging
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='path to image file')
    parser.add_argument('k', type=int, help='number of clusters')
    parser.add_argument('init', type=str, help='methods of initialization')
    parser.add_argument('--debug',
                        action='store_true',
                        help='show debug message')

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    filename = args.filename
    k = args.k
    init = args.init

    raw_filename = osp.splitext(filename)[0]
    gif_filename = 'images/%s_%d_%s_kkm_result.gif' % (raw_filename, k, init)
    png_filename = 'images/%s_%d_%s_kkm_result.png' % (raw_filename, k, init)

    data = load_data(filename)
    logger.debug('calculating gram matrix')
    gram = gram_matrix(data.reshape(-1, 5))

    logger.debug('initializing kernel k-means')
    kkm = KernelKMeans(data.reshape(-1, 5), k, gram, init)

    kkm.train()

    logger.debug('visualizing result')
    kkm.visualize_as_gif(data.shape[:2], gif_filename)
    kkm.visualize_as_image(data.shape[:2], png_filename)