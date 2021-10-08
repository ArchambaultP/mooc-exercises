from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    
    mid = int(shape[1] / 2)
    corner_h = int(mid/4)
        
    kernel = get_kernel(res[:,:mid])
    
    res[:,:mid] = kernel + 0.2
    res[-mid:, 0:corner_h] += 0.5
    
    res[:,mid:] = - kernel * 0.5 - 0.1
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one

    mid = int(shape[1] / 2)
    corner_h = int(mid/4)
    
    kernel = get_kernel(res[:,:mid])
    
    res[:,mid:] = kernel + 0.2
    res[-mid:,-corner_h:] += 0.5
    
    res[:,:mid] = - kernel * 0.5 - 0.1
    return res

def get_kernel(img):
    n_y, n_x = img.shape
    
    means = np.array([n_x,n_y], dtype=np.int) / 2
    cov = np.array([[1,0],[0,1]])
    # np.random.seed(2)
    points = np.random.multivariate_normal(means, cov, 100000)
    points[:,0] *= 4
    points[:,1] *= 2
    
    x_bins = n_x
    y_bins = n_y
    
    H, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=[x_bins, y_bins])
    H = H.T
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131, title='imshow: square bins')

    plt.imshow(H, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    H[np.where(H > 0)] = 1
    # print(H[int(n_y/2), :])
    
    return H
