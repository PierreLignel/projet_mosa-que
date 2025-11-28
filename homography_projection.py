
def homography_projection(I1, I2, x, y):
    h, w = I1.shape[:2]
    x1 = [0, w-1, w-1, 0]
    y1 = [0, 0, h-1, h-1]
    
    H = homography_estimate( x, y,x1, y1)

    for i in range(I2.shape[0]):
        for j in range(I2.shape[1]):
            x2, y2 = homography_apply(H, i, j)
            x2 = int(round(x2))
            y2 = int(round(y2))
            
            if 0 <= x2 < w and 0 <= y2 < h:
                I2[j][i] = I1[y2][x2]
    
    return I1

