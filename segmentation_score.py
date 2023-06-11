import cv2
import numpy as np
import os

def calcola_metriche(veri_positivi, falsi_positivi, falsi_negativi, veri_negativi):
    # Calcola la precisione
    precision = veri_positivi / (veri_positivi + falsi_positivi)

    # Calcola il recall
    recall = veri_positivi / (veri_positivi + falsi_negativi)

    # Calcola l'F1-score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calcola l'accuratezza
    accuracy = (veri_positivi + veri_negativi) / (veri_positivi + veri_negativi + falsi_positivi + falsi_negativi)

    return precision, recall, f1_score, accuracy

veri_positivi=0
veri_negativi=0
falsi_positivi=0
falsi_negativi=0
i=0
cartella1 = r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\maschere"
cartella2 = r"C:\Users\Sabino\Desktop\sistemi multimediali\database_sclere\maschere_tesiInglese"

# Ottieni la lista dei file nelle due cartelle
file_cartella1 = os.listdir(cartella1)
file_cartella2 = os.listdir(cartella2)

# Verifica che le due cartelle abbiano lo stesso numero di file
if len(file_cartella1) != len(file_cartella2):
    print("Le cartelle non contengono lo stesso numero di file.")
    exit()

# Cicla tra i file delle due cartelle in parallelo
for file1, file2 in zip(file_cartella1, file_cartella2):
    i+=1
    percorso_file1 = os.path.join(cartella1, file1)
    percorso_file2 = os.path.join(cartella2, file2)

    maschera1 = cv2.imread(percorso_file1, cv2.IMREAD_GRAYSCALE)
    maschera2 = cv2.imread(percorso_file2, cv2.IMREAD_GRAYSCALE)

    # Verifica se le maschere sono state caricate correttamente
    if maschera1 is None or maschera2 is None:
        print("Impossibile caricare le maschere.")
        exit()

    # Converti le maschere in array di booleani
    maschera1 = maschera1.astype(bool)
    maschera2 = maschera2.astype(bool)

    veri_positivi += np.sum(np.logical_and(maschera1, maschera2))
    falsi_positivi += np.sum(np.logical_and(maschera1, ~maschera2))
    veri_negativi += np.sum(np.logical_and(~maschera1, ~maschera2))
    falsi_negativi += np.sum(np.logical_and(~maschera1, maschera2))

# Stampa i risultati
print("Veri positivi:", veri_positivi)
print("Falsi positivi:", falsi_positivi)
print("Veri negativi:", veri_negativi)
print("Falsi negativi:", falsi_negativi)

precision, recall, f1_score, accuracy = calcola_metriche(veri_positivi, falsi_positivi, falsi_negativi, veri_negativi)

# Stampa i risultati
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("Accuracy:", accuracy)
print("Immagini analizzate:", i)