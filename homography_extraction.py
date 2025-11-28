import numpy as np
import matplotlib.pyplot as plt

def homography_extraction(I1, x, y, w, h):
    x2 = np.array([0, w-1, w-1, 0])
    y2 = np.array([0, 0, h-1, h-1])
    
    H = homography_estimate(x2, y2, x, y)
    
    I2 = np.zeros(h,w)
    
    x_rect = np.arange(0, w)
    y_rect = np.arange(0, h)
    
    for i in range(len(x_rect)):
        for j in range(len(y_rect)):
            (x_1f, y_1f) = homography_apply(H, x_rect[i], y_rect[j])
            I2[j][i] = I1[round(y_1f)][round(x_1f)]
    
    return I2

