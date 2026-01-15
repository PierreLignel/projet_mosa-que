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
    B2 = np.copy(mib[back])
    tab_coins = [[mib[back][0][0],mib[back][0][1]], [mib[back][1][0],mib[back][0][1]], [mib[back][1][0],mib[back][1][1]], [mib[back][0][0],mib[back][1][1]]]
    for i in range(4):
        (x, y) = homography_apply(H,tab_coins[i][0],tab_coins[i][1])
        tab_coins[i][0] = x
        tab_coins[i][1] = y
    xs = [p[0] for p in tab_coins]
    ys = [p[1] for p in tab_coins]
    B2[0][0] = min(xs)
    B2[0][1] = min(ys)
    B2[1][0] = max(xs)
    B2[1][1] = max(ys)

    dim_x = int(round(max(xs) - min(xs)))
    dim_y = int(round(max(ys) - min(ys)))

    mib2[back] = B2

    H_inv = np.linalg.inv(H)
    M2 = np.zeros((dim_y, dim_x), dtype=bool)
    if len(mib[image].shape) == 3:   # RGB
        C = mib[image].shape[2]
        mib2[image] = np.zeros((dim_y, dim_x, C), dtype=mib[image].dtype)
    else:                             # gris
        mib2[image] = np.zeros((dim_y, dim_x), dtype=mib[image].dtype)

        
    for yy in range(dim_y):
        for xx in range(dim_x):
            x2, y2 = homography_apply(H_inv, xx+mib2[back][0][0], yy+mib2[back][0][1])
            x2 = int(round(x2))
            y2 = int(round(y2))
            if 0 <= x2 < w and 0 <= y2 < h:
                if (mib[mask][y2][x2] == True):
                    mib2[image][yy][xx] = mib[image][y2][x2]
                    M2[yy][xx] = True
                else:
                    M2[yy][xx] = False
    mib2[mask] = M2
    return mib2

def MIB_fusion(mib_list):
    assert len(mib_list) > 0, "La liste de MIB est vide"

    # --- 1. Fusion des bounding boxes ---
    x_min = min(mib[back][0][0] for mib in mib_list)
    y_min = min(mib[back][0][1] for mib in mib_list)
    x_max = max(mib[back][1][0] for mib in mib_list)
    y_max = max(mib[back][1][1] for mib in mib_list)

    # On s'assure que ce sont des entiers pour les dimensions
    H = int(round(y_max - y_min + 1))
    W = int(round(x_max - x_min + 1))
    
    # On utilise le premier MIB de la liste pour les propriétés de base
    C = mib_list[0][image].shape[2]   
    dtype_out = mib_list[0][image].dtype

    # --- 2. Allocation du MIB résultat ---
    mib_res = np.empty(3, dtype=object)
    mib_res[back] = [[x_min, y_min], [x_max, y_max]]
    mib_res[mask] = np.zeros((H, W), dtype=bool)
    # On travaille toujours en float32 pendant le calcul de la moyenne
    mib_res[image] = np.zeros((H, W, C), dtype=np.float32)

    # --- 3. Fusion pixel par pixel ---
    for y in range(H):
        for x in range(W):
            pixels = []
            for m in mib_list:
                # Calcul des coordonnées locales dans chaque MIB
                yl = int(round(y + y_min - m[back][0][1]))
                xl = int(round(x + x_min - m[back][0][0]))

                # Vérification si le pixel est dans les limites du masque et s'il est True
                if (0 <= yl < m[mask].shape[0] and
                    0 <= xl < m[mask].shape[1] and
                    m[mask][yl, xl]):
                    
                    pixels.append(m[image][yl, xl].astype(np.float32))

            if len(pixels) > 0:
                mib_res[mask][y, x] = True
                # Moyenne de tous les pixels collectés
                mib_res[image][y, x] = np.mean(pixels, axis=0)

    # --- 4. Conversion finale (Correction du bug de couleur) ---
    if np.issubdtype(dtype_out, np.integer):
        # Si c'était du uint8 (0-255), on arrondit et on clip
        mib_res[image] = np.round(mib_res[image]).clip(0, 255).astype(dtype_out)
    else:
        # Si c'était du float (0.0-1.0), on garde tel quel (on ne divise pas par 255 !)
        mib_res[image] = mib_res[image].astype(dtype_out)

    return mib_res
    

img = plt.imread('qr-code-wall.png')
img2 = plt.imread('6113e94bc2096_les-femmes-sexposent-expo-exterieur.jpg')
img_rect = plt.imread('rectangle.jpg')
img3 = np.copy(img2)
w = 500
h = 500

#Selection des trois images
# Affiche l'image et clique 4 points
plt.imshow(img3, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 1")
points = plt.ginput(4)
plt.close()

x1 = np.array([p[0] for p in points])
y1 = np.array([p[1] for p in points])

plt.imshow(img3, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 2")
points2 = plt.ginput(4)
plt.close()

x2 = np.array([p[0] for p in points2])
y2 = np.array([p[1] for p in points2])

plt.imshow(img3, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 3")
points = plt.ginput(4)
plt.close()

x3 = np.array([p[0] for p in points])
y3 = np.array([p[1] for p in points])

Irect1 = homography_extraction(img3, x1, y1, w, h)
Irect2 = homography_extraction(img3, x2, y2, w, h)
Irect3 = homography_extraction(img3, x3, y3, w, h)


#On prend les points en commun a chaque fois
plt.imshow(Irect1, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 12")
points = plt.ginput(4)
plt.close()

xh1 = np.array([p[0] for p in points])
yh1 = np.array([p[1] for p in points])

plt.imshow(Irect2, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 12")
points = plt.ginput(4)
plt.close()

xh2 = np.array([p[0] for p in points])
yh2 = np.array([p[1] for p in points])

plt.imshow(Irect2, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 23")
points = plt.ginput(4)
plt.close()

xh3 = np.array([p[0] for p in points])
yh3 = np.array([p[1] for p in points])

plt.imshow(Irect3, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 23")
points = plt.ginput(4)
plt.close()

xh4 = np.array([p[0] for p in points])
yh4 = np.array([p[1] for p in points])


H21 = homography_estimate(xh2, yh2, xh1, yh1)
H32 = homography_estimate(xh4, yh4, xh3, yh3)



mib1 = MIB(Irect1)
mib2 = MIB(Irect2)
mib3 = MIB(Irect3)

mib2_t = MIB_transform(mib2, H21)
H31 = H21 @ H32 # On combine les deux matrices (3->2 puis 2->1)
mib3_t = MIB_transform(mib3, H31)
mib_list = []
mib_list.append(mib1)
mib_list.append(mib2_t)
mib_list.append(mib3_t)



mib_ff = MIB_fusion(mib_list)



plt.imshow(mib_ff[image], cmap='gray')  
plt.title("srtgf")
plt.show()





