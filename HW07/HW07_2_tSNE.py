# ---------------------------------------------------------------------------- #
# IOC5191 Machine Learning Homework 07-2: t-SNE
# Student ID: 309553002
# ---------------------------------------------------------------------------- #
import argparse
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

from tsne import pca, x2p, tsne

# ------------------------------- Symmetric SNE ------------------------------ #


def symmetric_sne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, _) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    C_history = []
    for iter in range(max_iter):

        # Compute pairwise affinities
        num = np.exp(-cdist(Y, Y, 'sqeuclidean'))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            C_history.append(C)

        # convergence condition
        if len(C_history) > 10 and abs(C_history[-1] - C_history[-2]) < 1e-6:
            break

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

        # Return solution for each iteration
        yield Y, P, Q


# ------------------------------- Main Function ------------------------------ #


def main(args):
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    methods = [tsne, symmetric_sne]
    method_names = ['tsne', 'ssne']

    if args.enable_part_1 or args.enable_part_2 or args.enable_part_3:
        for method, name in zip(methods, method_names):
            title = '%s_%d' % (name, args.perplexity)
            fig = plt.figure()
            frames = []
            Y = np.array([])
            P = np.array([])
            Q = np.array([])
            for idx, (Y, P, Q) in enumerate(method(X, 2, 50, args.perplexity)):
                if args.enable_part_2 and idx % 3 == 0:
                    frames.append((plt.scatter(Y[:, 0], Y[:, 1], 20,
                                               labels), ))

            if args.enable_part_2:
                interval = int(10000 / len(frames))
                ani = animation.ArtistAnimation(fig,
                                                frames,
                                                interval=interval,
                                                repeat=False)
                ani.save('images/%s.gif' % title, writer='pillow')

            if args.enable_part_1:
                fig = plt.figure()
                plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
                plt.savefig('images/%s.png' % title, dpi=500)

            if args.enable_part_3:
                # reordering
                indices = np.argsort(labels).flatten()
                P = P[indices]
                P = P[:, indices]
                # setting maximum color range as
                # [np.min(P), np.quantile(P, 0.985)]
                P[P >= np.quantile(P, 0.985)] = np.quantile(P, 0.985)
                fig = plt.figure('High Dimension Pairwise Similarity')
                plt.imsave('images/%s_hd.png' % title, P, cmap='hot')

                # reordering
                Q = Q[indices]
                Q = Q[:, indices]
                # setting maximum color range as
                # [np.min(P), np.quantile(P, 0.985)]
                Q[Q >= np.quantile(Q, 0.985)] = np.quantile(Q, 0.985)
                fig = plt.figure('Low Dimension Pairwise Similarity')
                plt.imsave('images/%s_ld.png' % title, Q, cmap='hot')


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('perplexity', type=int, default=20)
    parser.add_argument('--enable-part-1', action='store_true')
    parser.add_argument('--enable-part-2', action='store_true')
    parser.add_argument('--enable-part-3', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)
