import cv2
import numpy as np
import segmentation
from PIL import Image


def correctMask(mask):
    cv2.imshow("maschera", mask)
    # Converte la maschera in un array numpy
    mask_array = np.array(mask)

    # Specifica la coordinata y per la quale si desidera impostare a 0
    y_threshold = 330

    # Imposta a 0 tutti i valori della maschera sotto la coordinata y_threshold
    mask_array[y_threshold:, :] = 0

    return mask_array


# Read the image
image = cv2.imread('T_15_20190608_075438.jpg')

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
mask_array = correctMask(img_threshold)

# Applica la maschera all'immagine
masked_image_array = np.copy(image_array)
masked_image_array[mask_array == 0] = 0  # Imposta i pixel corrispondenti alla maschera a 0

# Crea un'immagine PIL dalla matrice dell'immagine mascherata
masked_image = Image.fromarray(masked_image_array)

# Mostra l'immagine mascherata
masked_image.show()

cv2.waitKey(0)
cv2.destroyAllWindows()