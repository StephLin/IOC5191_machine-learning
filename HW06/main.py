import subprocess

# Kernel K-Means
# for img in ['image1.png', 'image2.png']:
#     for init in ['random', 'k-means++']:
#         for k in map(str, [2, 3, 4]):
#             cmd = ([
#                 'python3', 'HW06_KernelKMeans.py', img, k, init, '--debug'
#             ])
#             print('$', ' '.join(cmd))
#             subprocess.check_output(cmd)

# Spectral Clustering
for img in ['image1.png', 'image2.png']:
    for init in ['random', 'k-means++']:
        for gl in ['unnormalized', 'normalized']:
            for k in map(str, [2, 3, 4]):
                cmd = ([
                    'python3', 'HW06_SpectralClustering.py', img, k, gl, init,
                    '--debug', '--cache'
                ])
                print('$', ' '.join(cmd))
                subprocess.check_output(cmd)
