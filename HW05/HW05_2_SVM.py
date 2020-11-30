# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 05-2: SVM
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import time
import enum
import multiprocessing as mp
import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

import libsvm.python.svmutil as svmutil

NUMBER_OF_CPUS = mp.cpu_count()

# ------------------------------- LIBSVM Params ------------------------------ #


class KernelType(enum.Enum):
    LINEAR = '-t 0'
    POLYNOMIAL = '-t 1'
    RADIAL_BASIS_FUNCTION = '-t 2'
    PRECOMPUTED_KERNEL = '-t 4'


# ----------------------------------- MNIST ---------------------------------- #


def load_data(image_filename, label_filename):
    images = np.genfromtxt(image_filename, delimiter=',')
    labels = np.genfromtxt(label_filename, delimiter=',').flatten()
    return images, labels


# ---------------------------- Precomputed kernel ---------------------------- #


def calc_feature_distance(x1, x2):
    return cdist(x1, x2, 'sqeuclidean')


def linear_plus_rbf_kernel(x1, x2, feature_distance, gamma):
    linear = x1 @ x2.T
    # https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    rbf = np.exp(-gamma * feature_distance)
    indices = np.arange(x1.shape[0])[:, np.newaxis] + 1
    param = np.hstack((indices, linear + rbf))
    return param


def train_and_predict(train_label, train_kernel, test_label, test_kernel,
                      train_param):
    model = svmutil.svm_train(train_label, train_kernel, param)
    _, accuracy, _ = svmutil.svm_predict(test_label, test_kernel, model)
    return accuracy[0]


# ------------------------------- Main function ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-part1', action='store_true')
    parser.add_argument('--disable-part2', action='store_true')
    parser.add_argument('--disable-part3', action='store_true')

    args = parser.parse_args()

    train_image, train_label = load_data('X_train.csv', 'Y_train.csv')
    test_image, test_label = load_data('X_test.csv', 'Y_test.csv')

    if not args.disable_part1:
        print('\nPART I')
        print('-' * 66)

        for ktype in KernelType:
            if ktype == KernelType.PRECOMPUTED_KERNEL:
                continue
            param = ktype.value + ' -q'
            start = time.perf_counter()
            model = svmutil.svm_train(train_label, train_image, param)
            end = time.perf_counter()
            duration = end - start

            _, accuracy, _ = svmutil.svm_predict(test_label, test_image, model)
            print('[%s] accuracy: %.2f%%, mse: %.2f, time: %.2f sec' %
                  (ktype.name, accuracy[0], accuracy[1], duration))

    if not args.disable_part2:
        print('\nPART II')
        print('-' * 66)

        ktype = KernelType.RADIAL_BASIS_FUNCTION

        cs = np.exp(np.arange(-5, 3))
        gammas = np.exp(np.arange(-9, 0))
        accuracy_matrix = np.zeros(cs.shape + gammas.shape)

        for i, c in enumerate(cs):
            args_list = []
            for j, gamma in enumerate(gammas):
                param = '-q %s -v 3 -s 0 -c %f -g %f' % (ktype.value, c, gamma)
                svm_args = (train_label, train_image, param)
                args_list.append(svm_args)

            with mp.Pool(NUMBER_OF_CPUS) as pool:
                results = pool.starmap(svmutil.svm_train, args_list)

            accuracy_matrix[i] = results

        plt.figure('PART II')
        sns.heatmap(accuracy_matrix,
                    xticklabels=np.log(gammas),
                    yticklabels=np.log(cs),
                    annot=True)
        plt.xlabel('gamma (log)')
        plt.ylabel('c (log)')

    if not args.disable_part3:
        print('\nPART III')
        print('-' * 66)

        ktype = KernelType.PRECOMPUTED_KERNEL
        gammas = np.exp(np.arange(-9, 0))
        args_list = []

        train_feature_distance = calc_feature_distance(train_image,
                                                       train_image)
        test_feature_distance = calc_feature_distance(test_image, test_image)

        for j, gamma in enumerate(gammas):
            param = '-q %s' % (ktype.value)
            train_kernel = linear_plus_rbf_kernel(train_image, train_image,
                                                  train_feature_distance,
                                                  gamma)
            test_kernel = linear_plus_rbf_kernel(test_image, test_image,
                                                 test_feature_distance, gamma)

            svm_args = (train_label, train_kernel, test_label, test_kernel,
                        param)
            args_list.append(svm_args)

        with mp.Pool(NUMBER_OF_CPUS) as pool:
            accuracy_matrix = pool.starmap(train_and_predict, args_list)

        accuracy_matrix = np.array([accuracy_matrix])

        plt.figure('PART III')
        sns.heatmap(accuracy_matrix, xticklabels=np.log(gammas), annot=True)
        plt.title('Linear + RBF')
        plt.xlabel('gamma (log)')

    if not args.disable_part2 or not args.disable_part3:
        plt.show()
