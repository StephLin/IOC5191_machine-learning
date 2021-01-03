# Homework 7

- Student ID: 309553002
- Name: 林育愷

## Prerequisites

Python 3.6^ involving following packages:

- `numpy`
- `scipy`
- `matplotlib`

## Usage

### 1. Kernel Eigenfaces

```txt
$ python3 HW07_1_KernelEigenFaces.py --help
usage: HW07_1_KernelEigenFaces.py [-h] [--enable-part-1] [--enable-part-2]
                                  [--enable-part-3] [--debug]

optional arguments:
  -h, --help       show this help message and exit
  --enable-part-1
  --enable-part-2
  --enable-part-3
  --debug
```

| Eigenfaces                                                   | Fisherfaces                                                    |
| ------------------------------------------------------------ | -------------------------------------------------------------- |
| ![eigenfaces](images/eigenfaces.png)                         | ![fisherfaces](images/fisherfaces.png)                         |
| **Reconstruction of eigenfaces**                             | **Reconstruction of fisherfaces**                              |
| ![eigenfaces_reconstruct](images/eigenfaces_reconstruct.png) | ![fisherfaces_reconstruct](images/fisherfaces_reconstruct.png) |

| k   | PCA       | LDA       | Linear Kernel PCA | Linear Kernel LDA | Polynomial Kernel PCA $(c, d) = (1, 2)$ | Polynomial Kernel LDA $(c, d) = (1, 2)$ | RBF Kernel PCA $\gamma=10^{-3}$ | RBF Kernel LDA $\gamma=10^{-3}$ |
| --- | --------- | --------- | ----------------- | ----------------- | --------------------------------------- | --------------------------------------- | ------------------------------- | ------------------------------- |
| 3   | **96.67** | 63.33     | 83.33             | 86.67             | 83.33                                   | 86.67                                   | 83.33                           | 86.67                           |
| 5   | **96.67** | 70        | 83.33             | **90**            | **86.67**                               | **90**                                  | 80                              | 86.67                           |
| 7   | **96.67** | **73.33** | 80                | 86.67             | 83.33                                   | 86.67                                   | 80                              | 86.67                           |
| 9   | **96.67** | 66.67     | 83.33             | 83.33             | 83.33                                   | 86.67                                   | 83.33                           | 86.67                           |
| 11  | **96.67** | 66.67     | **86.67**         | 86.67             | **86.67**                               | 86.67                                   | **86.67**                       | 86.67                           |
| 13  | **96.67** | 66.67     | 80                | 86.67             | 76.67                                   | 86.67                                   | 76.67                           | **90**                          |
| 15  | **96.67** | 63.33     | 80                | 83.33             | 76.67                                   | 86.67                                   | 80                              | **90**                          |

### 2. t-SNE

You can run single task (one perplexity value) using the following command:

```txt
$ python3 HW07_2_tSNE.py --help
usage: HW07_2_tSNE.py [-h] [--enable-part-1] [--enable-part-2]
                      [--enable-part-3] [--debug]
                      perplexity

positional arguments:
  perplexity

optional arguments:
  -h, --help       show this help message and exit
  --enable-part-1
  --enable-part-2
  --enable-part-3
  --debug
```

Or you are free to use `script.py` to run several tasks:

```txt
python3 script.py
```

|                   | t-SNE                                                | Symmetric SNE                                        |
| ----------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| **perplexity=05** | [![tsne_5](images/tsne_5.png)](images/tsne_5.gif)    | [![ssne_5](images/ssne_5.png)](images/ssne_5.gif)    |
| **perplexity=10** | [![tsne_10](images/tsne_10.png)](images/tsne_10.gif) | [![ssne_10](images/ssne_10.png)](images/ssne_10.gif) |
| **perplexity=15** | [![tsne_15](images/tsne_15.png)](images/tsne_15.gif) | [![ssne_15](images/ssne_15.png)](images/ssne_15.gif) |
| **perplexity=20** | [![tsne_20](images/tsne_20.png)](images/tsne_20.gif) | [![ssne_20](images/ssne_20.png)](images/ssne_20.gif) |
| **perplexity=25** | [![tsne_25](images/tsne_25.png)](images/tsne_25.gif) | [![ssne_25](images/ssne_25.png)](images/ssne_25.gif) |
| **perplexity=30** | [![tsne_30](images/tsne_30.png)](images/tsne_30.gif) | [![ssne_30](images/ssne_30.png)](images/ssne_30.gif) |

|                   | t-SNE High Dimension Pairwise Similarities | t-SNE Low Dimension Pairwise Similarities | Symmetric SNE High Dimension Pairwise Similarities | Symmetric SNE Low Dimension Pairwise Similarities |
| ----------------- | ------------------------------------------ | ----------------------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| **perplexity=05** | ![tsne_5](images/tsne_5_hd.png)            | ![tsne_5](images/tsne_5_ld.png)           | ![ssne_5](images/ssne_5_hd.png)                    | ![ssne_5](images/ssne_5_ld.png)                   |
| **perplexity=10** | ![tsne_10](images/tsne_10_hd.png)          | ![tsne_10](images/tsne_10_ld.png)         | ![ssne_10](images/ssne_10_hd.png)                  | ![ssne_10](images/ssne_10_ld.png)                 |
| **perplexity=15** | ![tsne_15](images/tsne_15_hd.png)          | ![tsne_15](images/tsne_15_ld.png)         | ![ssne_15](images/ssne_15_hd.png)                  | ![ssne_15](images/ssne_15_ld.png)                 |
| **perplexity=20** | ![tsne_20](images/tsne_20_hd.png)          | ![tsne_20](images/tsne_20_ld.png)         | ![ssne_20](images/ssne_20_hd.png)                  | ![ssne_20](images/ssne_20_ld.png)                 |
| **perplexity=25** | ![tsne_25](images/tsne_25_hd.png)          | ![tsne_25](images/tsne_25_ld.png)         | ![ssne_25](images/ssne_25_hd.png)                  | ![ssne_25](images/ssne_25_ld.png)                 |
| **perplexity=30** | ![tsne_30](images/tsne_30_hd.png)          | ![tsne_30](images/tsne_30_ld.png)         | ![ssne_30](images/ssne_30_hd.png)                  | ![ssne_30](images/ssne_30_ld.png)                 |
