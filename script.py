import os
import cv2
# Resize the image
new_size = (1200, 1600)

# Definisci il percorso della directory principale
directory_principale = r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\Italiano congiuntive\Dataset congiuntive italiano segmentato"

cartella_destinazione = r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\maschere_tesiInglese OS"

# Cicla attraverso ogni elemento nella directory principale
for elemento in os.listdir(directory_principale):
    # Crea il percorso completo dell'elemento
    percorso_elemento = os.path.join(directory_principale, elemento)
    for file in os.listdir(percorso_elemento):
        # Verifica se il file è l'immagine giusta
        if file.endswith("sclera.png"):
            # Crea il percorso completo del file
            percorso_file = os.path.join(percorso_elemento, file)

            # Apri l'immagine utilizzando cv2
            image = cv2.imread(percorso_file)
            resized_image = cv2.resize(image, new_size)

            # Applica maschera
            mask = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)

            # Salva l'immagine nella cartella di destinazione
            nome_file = os.path.basename(percorso_file)
            percorso_file = str(cartella_destinazione) +"/"+ str(nome_file) +".jpg"
            cv2.imwrite(percorso_file, mask)