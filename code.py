import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import io, color, feature


def homography_estimate(x1, y1, x2, y2):
    
    A = np.zeros((8, 8))
    B = np.zeros(8)
    H = np.zeros((3, 3))
    
    for i in range(4):
        A[2*i][0] = x1[i]
        A[2*i][1] = y1[i]
        A[2*i][2] = 1;
        A[2*i][6] = -(x1[i]*x2[i])
        A[2*i][7] = -(y1[i]*x2[i])
        A[(2*i)+1][3] = x1[i]
        A[(2*i)+1][4] = y1[i]
        A[(2*i)+1][5] = 1
        A[(2*i)+1][6] = -(x1[i]*y2[i])
        A[(2*i)+1][7] = -(y1[i]*y2[i])
    
    for i in range(4):
        B[2*i] = x2[i]
        B[(2*i)+1] = y2[i]
        
    X = np.linalg.solve(A, B)
    
    H[0][0] = X[0]
    H[0][1] = X[1]
    H[0][2] = X[2]
    H[1][0] = X[3]
    H[1][1] = X[4]
    H[1][2] = X[5]
    H[2][0] = X[6]
    H[2][1] = X[7]
    H[2][2] = 1
    
    return H



def homography_apply(H, x1, y1):
    x2 = (H[0][0]* x1 + H[0][1]* y1 + H[0][2])/(H[2][0]* x1 + H[2][1]* y1 + H[2][2])
    y2 = (H[1][0]* x1 + H[1][1]* y1 + H[1][2])/(H[2][0]* x1 + H[2][1]* y1 + H[2][2])

    return x2, y2


x1 = np.array([10, 40, 60, 100])
y1 = np.array([40, 100, 10, 60])
x2 = np.array([25, 25, 75, 75])
y2 = np.array([25, 75, 25, 75])


H = homography_estimate(x1, y1, x2, y2)

x1_tot = np.arange(0, 101)
y1_tot = np.arange(0, 101)

(x2_f, y2_f) = homography_apply(H, x1, y1)

print(H)
print(x2_f)
print(y2_f)




