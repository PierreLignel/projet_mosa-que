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

def MIBFusion(mib1, mib2):
    # --- Calcul du back (bounding box fusionné) ---
    x_min = min(mib1[back][0][0], mib2[back][0][0])
    y_min = min(mib1[back][0][1], mib2[back][0][1])
    x_max = max(mib1[back][1][0], mib2[back][1][0])
    y_max = max(mib1[back][1][1], mib2[back][1][1])

    # Dimensions du MIB résultat
    H = y_max - y_min + 1
    W = x_max - x_min + 1

    # --- Allocation du MIB résultat ---
    mib = np.empty(3, dtype=object)
    mib[back]  = [[x_min, y_min], [x_max, y_max]]
    mib[mask]  = np.zeros((H, W), dtype=bool)

    
    
        
    C = mib1[image].shape[2]
    mib[image] = np.zeros((H, W, C), dtype=mib1[image].dtype)

    # --- Fusion pixel par pixel ---
    for y in range(H):
        for x in range(W):
            # Coordonnées locales dans mib1
            y1 = y + y_min - mib1[back][0][1]
            x1 = x + x_min - mib1[back][0][0]

            # Coordonnées locales dans mib2
            y2 = y + y_min - mib2[back][0][1]
            x2 = x + x_min - mib2[back][0][0]

            # Masques valides ?
            m1_valid = (0 <= y1 < mib1[mask].shape[0]) and (0 <= x1 < mib1[mask].shape[1])
            m2_valid = (0 <= y2 < mib2[mask].shape[0]) and (0 <= x2 < mib2[mask].shape[1])
            if m1_valid :
                m1 = mib1[mask][y1, x1]
            else:
                m1 = False
            if m2_valid :
                m2 = mib2[mask][y2, x2]
            else:
                m2 = False
            

            if m1 or m2:
                mib[mask][y, x] = True

                if m1 and m2:
                    # moyenne des pixels
                    mib[image][y, x] = (mib1[image][y1, x1].astype(np.float32) + 
                                        mib2[image][y2, x2].astype(np.float32)) / 2
                    # si nécessaire, reconvertir en uint8
                    if mib[image].dtype == np.uint8:
                        mib[image][y, x] = np.round(mib[image][y, x]).astype(np.uint8)
                elif m1:
                    mib[image][y, x] = mib1[image][y1, x1]
                else:
                    mib[image][y, x] = mib2[image][y2, x2]

    return mib


    

img = plt.imread('challenge1.png')

if img.ndim == 2:
    height, width = img.shape
    white_rect = np.ones((500, width))
else:
    height, width, channels = img.shape
    white_rect = np.ones((500, width, channels))

# Concaténer le rectangle noir au-dessus de l'image
new_img = np.vstack((white_rect, img))

# Affiche l'image et clique 4 points
plt.imshow(new_img, cmap='gray')  
plt.title("Clique 4 points dans l'ordre souhaité 1")
points = plt.ginput(4)
plt.close()

x = np.array([p[0] for p in points])
y = np.array([p[1] for p in points])
w = 500
h = 500

Irect = homography_extraction(new_img, x, y, w, h)
# 1. Récupération des données
data = np.array(Irect)

# --- BLOC DE DIAGNOSTIC (Affiche les infos pour comprendre) ---
print(f"Type de données : {data.dtype}")
print(f"Valeur min : {data.min()}, Valeur max : {data.max()}")
print(f"Forme de l'image : {data.shape}")
# --------------------------------------------------------------

# 2. CORRECTION AUTOMATIQUE 0-1 vs 0-255
# Si le maximum est petit (<= 1.0), c'est que l'image est en float 0-1.
# On la remet en 0-255.
if data.max() <= 1.0:
    print(">> Correction : Conversion de l'image de 0-1 vers 0-255")
    data = (data * 255).astype(np.int16)
else:
    # Sinon on s'assure juste d'avoir des entiers signés pour la soustraction
    data = data.astype(np.int16)

# 3. GESTION DU 4ème CANAL (Transparence/Alpha)
if data.shape[-1] == 4:
    data = data[:, :, :3]

# Définition des couleurs
colors = {
    "rouge": [255, 0, 0],
    "jaune": [255, 255, 0],
    "vert": [0, 255, 0],
    "magenta": [255, 0, 255],
    "rose": [255, 192, 203],
    "cyan": [0, 255, 255],
    "bleu": [0, 0, 255],
    "blanc": [255, 255, 255], # Ajouté pour test
    "noir": [0, 0, 0]
}

tol = 50
pixel_counts = {}

# On parcourt chaque pixel pour trouver la couleur LA PLUS PROCHE
# (Cette méthode est plus précise que ton masque précédent qui pouvait compter un pixel deux fois)

# Aplatir l'image en une liste de pixels (N, 3) pour aller plus vite
pixels = data.reshape(-1, 3)
nb_total_pixels = len(pixels)

print(f"Analyse de {nb_total_pixels} pixels...")

# Initialiser les compteurs à 0
for c in colors:
    pixel_counts[c] = 0
pixel_counts["inconnu"] = 0

# VERSION VECTORISÉE RAPIDE (Calcule la distance avec toutes les couleurs)
# On ne boucle pas sur les pixels (trop lent), on boucle sur les couleurs
distances = []
color_names = list(colors.keys())
color_values = np.array(list(colors.values())) # Forme (Nb_couleurs, 3)

# Pour chaque pixel, on calcule la distance vers chaque couleur de référence
# Astuce NumPy : On utilise broadcasting
# pixels[:, None, :] a la forme (N_pixels, 1, 3)
# color_values[None, :, :] a la forme (1, N_couleurs, 3)
# La diff nous donne (N_pixels, N_couleurs, 3)
# C'est lourd en mémoire si l'image est énorme. Si l'image est petite/moyenne (ex: < 1000x1000), ça passe.

# Si l'image est grande, on utilise une méthode plus simple :
# On calcule la distance pour chaque couleur séparément.
min_distances = np.full(len(pixels), 9999) # Distance min trouvée pour chaque pixel
closest_color_indices = np.full(len(pixels), -1) # Index de la couleur trouvée

for idx, (name, rgb) in enumerate(zip(color_names, color_values)):
    # Distance euclidienne (plus précise que la valeur absolue simple)
    # dist = sqrt((r1-r2)² + (g1-g2)² + (b1-b2)²)
    d = np.linalg.norm(pixels - rgb, axis=1)
    
    # Si cette couleur est plus proche que la précédente trouvée pour ce pixel
    better_mask = d < min_distances
    min_distances[better_mask] = d[better_mask]
    closest_color_indices[better_mask] = idx

# Maintenant on filtre ceux qui sont hors tolérance
valid_mask = min_distances <= tol

# Compter
import collections
found_indices = closest_color_indices[valid_mask]
counts = collections.Counter(found_indices)

for idx, count in counts.items():
    pixel_counts[color_names[idx]] = count

pixel_counts["inconnu"] = nb_total_pixels - np.sum(valid_mask)

# Affichage
sorted_counts = sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True)
for color, count in sorted_counts:
    if count > 0:
        print(f"{color}: {count} pixels")







