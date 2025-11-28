import numpy as np
import matplotlib.pyplot as plt

def homography_extraction(I1, x, y, w, h):
    x2 = np.array([0, w-1, w-1, 0])
    y2 = np.array([0, 0, h-1, h-1])
    
    H = homography_estimate(x2, y2, x, y)
    
    I2 = np.zeros(h,w)
    
    x_rect = np.arange(0, w)
    y_rect = np.arange(0, h)
    
    (x_1f, y_1f) = homography_apply(H, x_rect, y_rect)
    
    for i in range(len(x_1f)):
        for j in range(len(y_1f)):
            I2[i][j] = I1[round(x_1f[i])][round(y_1f[j])]
    
    return I2

