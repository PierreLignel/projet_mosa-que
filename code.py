import numpy as np
import matplotlib.pyplot as plt
mask = 0
image = 1
back  = 2


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
        # image en niveaux de gris
        I2 = np.zeros((h, w), dtype=I1.dtype)
    else:
        # image couleur RGB
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
    
    H = homography_estimate( x, y,x1, y1)

    for yy in range(I2.shape[0]):
        for xx in range(I2.shape[1]):
            x2, y2 = homography_apply(H, xx, yy)
            x2 = int(round(x2))
            y2 = int(round(y2))
            
            if 0 <= x2 < w and 0 <= y2 < h:
                I2[yy][xx] = I1[y2][x2]

    return I2
 
def homography_cross_projection(I1, x1, y1, x2, y2):
    h ,w = I1.shape[:2]
    h_carre = 100
    w_carre = 100
    I2 = np.zeros_like(I1)

    x_carre = [0, w_carre-1, w_carre-1, 0]
    y_carre = [0, 0, h_carre-1, h_carre-1]
    H1_c = homography_estimate(x1, y1,x_carre, y_carre)
    H2_c = homography_estimate(x2, y2,x_carre, y_carre)
    H1_2 = homography_estimate(x1, y1,x2, y2)
    H2_1 = homography_estimate(x2, y2,x1, y1)
    #carre_magique = np.zeros((h, w))
    for yy in range(I2.shape[0]):
        for xx in range(I2.shape[1]):
            x2, y2 = homography_apply(H1_c, xx, yy)
            x2 = int(round(x2))
            y2 = int(round(y2))
            
            
            
            x2_bis, y2_bis = homography_apply(H2_c, xx, yy)
            x2_bis = int(round(x2_bis))
            y2_bis = int(round(y2_bis))
            if 0 <= x2 < w_carre and 0 <= y2 < h_carre:
                xxx ,yyy =homography_apply(H1_2,xx,yy)
                xxx = int(round(xxx))
                yyy = int(round(yyy))
                
                I2[yy][xx] = I1[yyy][xxx]
            elif 0 <= x2_bis < w_carre and 0 <= y2_bis < h_carre:
                    xxx_bis ,yyy_bis =homography_apply(H2_1,xx,yy)
                    xxx_bis = int(round(xxx_bis))
                    yyy_bis = int(round(yyy_bis))
                    
                    I2[yy][xx] = I1[yyy_bis][xxx_bis]
            else:
                I2[yy][xx] = I1[yy][xx]

    return I2


def MIB(I):
    h, w = I.shape[:2]

    M = np.ones((h, w), dtype=bool)   # mask
    B = [[0,0], [w-1,h-1]]            #Borne

    mib = np.empty(3, dtype=object)   # tableau numpy qui contient 3 objets
    mib[mask] = M
    mib[image] = I
    mib[back] = B

    return mib

def MIB_transform(mib, H):
    mib2 = np.empty(3, dtype=object)

    h, w = mib[image].shape[:2]
    B2 = mib[back]
    tab_coins = [[mib[back][0][0],mib[back][0][1]], [mib[back][1][0],mib[back][0][1]], [mib[back][1][0],mib[back][1][1]], [mib[back][0][0],mib[back][1][1]]]
    for i in range(4):
        (x, y) = homography_apply(H,tab_coins[i][0],tab_coins[i][1])
        tab_coins[i][0] = x
        tab_coins[i][1] = y
    B2[0][0] = min(tab_coins[:][0])
    B2[0][1] = min(tab_coins[:][1])
    B2[1][0] = max(tab_coins[:][0]) 
    B2[1][1] = max(tab_coins[:][1])
    dim_x = max(tab_coins[:][0]) - min(tab_coins[:][0])
    dim_y = max(tab_coins[:][1]) - min(tab_coins[:][1])
    mib2[back] = B2

    H_inv = np.linalg.inv(H)
    M2 = np.zeros((dim_y, dim_x), dtype=bool)
    I2 = np.zeros((dim_y, dim_x))
    for yy in range(dim_y):
        for xx in range(dim_x):
            x2, y2 = homography_apply(H_inv, xx+mib2[back][0][0], yy+mib2[back][0][1])
            x2 = int(round(x2))
            y2 = int(round(y2))
            if 0 <= x2 < w and 0 <= y2 < h:
                if (mib[mask][y2][x2] == True)
                    mib2[image][yy][xx] = mib[image][y2][x2]
                    M2[yy][xx] = True
                else:
                    M2[yy][xx] = False
    mib2[mask] = M2
    return mib2

    

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


# Irect = homography_extraction(img3, x1, y1, w, h)
# I2 = homography_projection(Irect, img3, x2, y2)
I2 = homography_cross_projection(img3, x1, y1, x2, y2)

plt.imshow(I2, cmap='gray') 
plt.axis('off')              
plt.show()





