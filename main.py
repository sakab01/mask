import cv2
import numpy as np
import segmentation
import os
from PIL import Image

# Resize the image
new_size = (1200, 1600)

cartella_destinazione = r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\original size mask"

# Cicla su tutti i file nella cartella
for file in os.listdir(r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\Italiano congiuntive\Dataset congiuntive gruppo anemia  organizzato 28 mar 2020\Trasfusionale congiuntive"):
    # Verifica se il file è un'immagine
    if file.endswith(".jpg"):    
        
        # Crea il percorso completo del file
        percorso_file = os.path.join(r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\Italiano congiuntive\Dataset congiuntive gruppo anemia  organizzato 28 mar 2020\Trasfusionale congiuntive", file)
        
        # Apri l'immagine utilizzando cv2
        image = cv2.imread(percorso_file)

        resized_image = cv2.resize(image, new_size)

        img_threshold, kmeans, sclera_ncut, ncut = segmentation.segment(resized_image)

        # Applica la maschera all'immagine
        masked_image_array = np.copy(img_threshold)
        masked_image_array[img_threshold == 0] = 0  # Imposta i pixel corrispondenti alla maschera a 0

        # Crea un'immagine PIL dalla matrice dell'immagine mascherata
        masked_image = Image.fromarray(masked_image_array)
        
        nome_file = os.path.basename(percorso_file)

        # Salva l'immagine nella cartella di destinazione
        nome_file = nome_file+"_maschera"
        percorso_file = str(cartella_destinazione) +"/"+ str(nome_file) +".jpg"
        masked_image.save(percorso_file)
        