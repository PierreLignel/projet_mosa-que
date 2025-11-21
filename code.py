import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import io, color, feature


def homography_apply(H, x1, y1):
    x2 = H[0][0]* x1 + H[0][1]* y1 + H[0][2]/(H[2][0]* x1 + H[2][1]* y1 + H[2][2])
    y2 = H[1][0]* x1 + H[1][1]* y1 + H[1][2]/(H[2][0]* x1 + H[2][1]* y1 + H[2][2])

    return x2, y2
