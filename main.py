import cv2
import numpy as np
import segmentation
from scipy import ndimage
from PIL import Image


def calcola_area_e_filtra(image):
    
    # Etichettatura delle regioni connesse
    labeled_image, num_labels = ndimage.label(image)
    
    # Calcola le dimensioni delle regioni connesse
    sizes = ndimage.sum(image, labeled_image, range(1, num_labels + 1))
    
    # Trova l'indice dell'area più grande
    largest_area_index = np.argmax(sizes)
    
    # Crea una maschera per mantenere solo l'area più grande
    largest_area_mask = np.zeros_like(image)
    largest_area_mask[labeled_image == largest_area_index + 1] = 255
    
    # Mostra l'immagine originale
    cv2.imshow('Immagine originale', image)
    
    # Mostra la maschera dell'area più grande
    cv2.imshow('Area più grande', largest_area_mask)
    
    return largest_area_mask

# Path dell'immagine
#image = cv2.imread('T_15_20190608_075438.jpg')
image = cv2.imread('')

# Resize the image
new_size = (300, 400)

resized_image = cv2.resize(image, new_size)

img_threshold, kmeans, sclera_ncut, ncut = segmentation.segment(resized_image)

cv2.imshow("originale",resized_image)
img_threshold = np.where(img_threshold == 1, 255, img_threshold)
cv2.imshow("img_threshold",img_threshold)

# Converte l'immagine e la maschera in array numpy
image_array = np.array(resized_image)
#mask_array = np.array(img_threshold)
mask_array = calcola_area_e_filtra(img_threshold)

# Applica la maschera all'immagine
masked_image_array = np.copy(image_array)
masked_image_array[mask_array == 0] = 0  # Imposta i pixel corrispondenti alla maschera a 0

# Crea un'immagine PIL dalla matrice dell'immagine mascherata
masked_image = Image.fromarray(masked_image_array)

# Mostra l'immagine mascherata
masked_image.show()

cv2.waitKey(0)
cv2.destroyAllWindows()