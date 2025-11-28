import numpy as np
import matplotlib.pyplot as plt


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


def homography_extraction(I1, x, y, w, h):
    x2 = np.array([0, w-1, w-1, 0])
    y2 = np.array([0, 0, h-1, h-1])
    
    H = homography_estimate(x2, y2, x, y)
    
    if len(I1.shape) == 2:
        # Image en niveaux de gris
        I2 = np.zeros((h, w), dtype=I1.dtype)
    else:
        # Image couleur RGB
        channels = I1.shape[2]
        I2 = np.zeros((h, w, channels), dtype=I1.dtype)
    
    x_rect = np.arange(0, w)
    y_rect = np.arange(0, h)
    
    
    for i in range(len(x_rect)):
        for j in range(len(y_rect)):
            (x_1f, y_1f) = homography_apply(H, x_rect[i], y_rect[j])
            I2[j][i] = I1[round(y_1f)][round(x_1f)]
    
    return I2


def homography_projection(I1, I2, x, y):
    h, w = I1.shape[:2]
    x1 = [0, w-1, w-1, 0]
    y1 = [0, 0, h-1, h-1]
    
    H = homography_estimate(x, y, x1, y1)

    for i in range(I2.shape[0]):
        for j in range(I2.shape[1]):
            (x2, y2) = homography_apply(H, j, i)
            x2 = int(round(x2))
            y2 = int(round(y2))
            
            if 0 <= x2 < w and 0 <= y2 < h:
                I2[i][j] = I1[y2][x2]
    
    return I2


img = plt.imread('qr-code-wall.png')
img2 = plt.imread('6113e94bc2096_les-femmes-sexposent-expo-exterieur.jpg')
img_rect = plt.imread('rectangle.jpg')
img3 = np.copy(img2)

# Affiche l'image et clique 4 points
plt.imshow(img3, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité")
points = plt.ginput(4)
plt.close()

plt.imshow(img3, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité")
points2 = plt.ginput(4)
plt.close()


# Sépare les coordonnées x et y
x1 = np.array([p[0] for p in points])
y1 = np.array([p[1] for p in points])
x2 = np.array([p[0] for p in points2])
y2 = np.array([p[1] for p in points2])
w = 500
h = 500


Irect = homography_extraction(img3, x1, y1, w, h)
I2 = homography_projection(Irect, img3, x2, y2)

plt.imshow(I2, cmap='gray') 
plt.axis('off')              
plt.show()





