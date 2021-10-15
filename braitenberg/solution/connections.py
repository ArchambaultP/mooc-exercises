from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    
    mid = int(shape[1] / 2)
    quart = int(mid/2)
    corner_h = int(mid/4)
    
    t = np.linspace(0, shape[0],num=shape[0]).T
    t /= (shape[0] * 2)
    print(shape)
    h = np.linspace(0, shape[1],num=mid).T
    h /= (shape[1] * 2)
    
    grad_t = np.tile(t, (mid,1))
    grad_h = np.tile(h, (shape[0],1))
    
    grad = grad_t.T + grad_h    
    
    res[:,:mid] = grad_t.T
    res[:,mid:] = grad_t.T * 0.3
    
    res[:,:corner_h] = -0.2
    res[:,-corner_h:] = -0.1
    
    res[:,:mid] = grad * 0.8
    res[:,mid:] = np.flip(grad, axis=1) * 0.2
    
    
    return res * 2


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    
    mid = int(shape[1] / 2)
    corner_h = int(mid/4)
    

    t = np.linspace(0, shape[0],num=shape[0]).T
    t /= shape[0]
    h = np.linspace(0, shape[1],num=mid).T
    h /= (shape[1] * 2)
    
    grad_t = np.tile(t, (mid,1))
    grad_h = np.tile(h, (shape[0],1))
    grad = np.flip(grad_t.T + grad_h, axis=1)

    res[:,:mid] = grad_t.T * 0.3
    res[:,mid:] = grad_t.T
    h = int(shape[0]/2)
    res[:,:corner_h] = -0.1
    res[:,-corner_h:] = -0.2
    
    res[:,mid:] = grad * 0.8
    res[:,:mid] = np.flip(grad, axis=1) * 0.2
    
    return res

def get_kernel(img):
    n_y, n_x = img.shape
    
    means = np.array([n_x,n_y], dtype=np.int) / 2
    cov = np.array([[1,0],[0,1]])
    points = np.random.multivariate_normal(means, cov, 1000000)
    points[:,0] *= 4
    points[:,1] *= 2
    
    x_bins = n_x
    y_bins = n_y
    
    H, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=[x_bins, y_bins])
    H = H.T
    
    # fig = plt.figure(figsize=(7, 3))
    # ax = fig.add_subplot(131, title='imshow: square bins')
    # plt.imshow(H, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    H[np.where(H > 0)] = 1
    # print(H[int(n_y/2), :])
    
    return H

def goodTest():
    mid = int(shape[1] / 2)
    corner_h = int(mid/4)
    
    res[:,:mid] = 0.3
    res[:,mid:] = 0.7
    
    h = int(shape[0]/2)
    res[:h,:corner_h] = -0.1
    res[:h,-corner_h:] = -0.2