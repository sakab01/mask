import cv2
import numpy as np
from operator import itemgetter
import exposure
import norm_cuts as nc
from scipy import ndimage
from threshold import threshold, refine

SHOW_NCUT = True
WRITE_STEPS = False
VERBOSE = False

L_min = 160
L_max = 211
L_25 = 193
L_75 = 180
A_min = 130
A_max = 145
A_25 = 139
A_75 = 135
B_min = 109
B_max = 123
B_25 = 118
B_75 = 114

def segment(img, fix_range=0, cuts=10, compactness=10, 
    blur_scale=1, n_cuts=10, n_thresh=0.1,
    imp_cuts=10, imp_thresh=0.1, imp_comp=6, 
    imp_fix=30.0, gamma=1):
 
    retinex_img = exposure.automatedMSRCR(img)
    preprocessed = exposure.preprocess(retinex_img)
    original, kmeans, ncut = nc.nCut(preprocessed, cuts=cuts, 
    compactness=compactness, n_cuts=n_cuts, thresh=n_thresh)
    img_threshold = threshold(retinex_img)
    mask_ncut = nc.gaussian_mask(ncut, img_threshold)
    sclera_ncut, mask_ncut = nc.jointRegions(img, ncut, mask_ncut, fix_range, 0)

    img_threshold = np.where(img_threshold == 1, 255, img_threshold)
    
    # Converte l'immagine e la maschera in array numpy
    
    img_threshold = calcola_area_e_filtra(img_threshold)
    
    return img_threshold, kmeans, sclera_ncut, ncut

def calcola_area_e_filtra(image, soglia_y=380):
    # Etichettatura delle regioni connesse
    labeled_image, num_labels = ndimage.label(image)
    
    # Calcola le dimensioni delle regioni connesse
    sizes = ndimage.sum(image, labeled_image, range(1, num_labels + 1))
    
    # Applica il filtro di soglia y per scartare i pixel al di sotto della soglia
    filtered_image = np.where(image >= soglia_y, image, 0)
    
    # Ricalcola le dimensioni delle regioni connesse dopo l'applicazione del filtro di soglia
    filtered_sizes = ndimage.sum(filtered_image, labeled_image, range(1, num_labels + 1))
    
    # Trova l'indice dell'area più grande dopo l'applicazione del filtro di soglia
    largest_area_index = np.argmax(filtered_sizes)
    
    # Crea una maschera per mantenere solo l'area più grande
    largest_area_mask = np.zeros_like(image)
    largest_area_mask[labeled_image == largest_area_index + 1] = 255
    
    return largest_area_mask


def suspect(img):
    l, alpha, beta = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    alpha_avg = np.average(alpha[l != 0])
    beta_avg = np.average(beta[l != 0])
    lum_avg = np.average(l[l != 0])
 # if VERBOSE:
    print(lum_avg, alpha_avg, beta_avg)
    l_distance = max(lum_avg - L_75, lum_avg - L_25) / (L_max-L_min)
    a_distance = max(alpha_avg - A_75, alpha_avg - A_25) / (A_max-A_min)
    b_distance = max(beta_avg - B_75, beta_avg - B_25) / (B_max-B_min)
    score = np.average([l_distance, a_distance, b_distance])
    print([l_distance, a_distance, b_distance], score)
    return score

def improve_precision_ncut(original, img_threshold, seg_img, 
    mask, preprocessed,
    ret_img, res_img, max_iter=3, 
    blur_scale=1, cuts=20, comp=6, thresh=0.1, fix=30.0, gamma=0.8):
    if blur_scale != 1 or gamma != 0.8:
        preprocessed = exposure.preprocess(ret_img, blur_scale=blur_scale, gamma=gamma)
    if SHOW_NCUT:
        cv2.imshow('preprocessed', preprocessed)
        cv2.imshow('first_segment', seg_img)
        i = 0
        history = []
        while i < max_iter:
            original = original * np.where(mask == 255, 1, mask)
            suspect_score = suspect(original)
            history.append({
            'result': res_img,
            'segment': seg_img,
            'threshold': img_threshold,
            'mask': mask,
            'score': abs(suspect_score)})
            if abs(suspect_score) > 0.05:
                preprocessed = preprocessed * np.where(mask == 255, 1, mask)
                indexes = preprocessed == 0
                preprocessed = np.where(indexes, 240, preprocessed)
                preprocessed, kmeans, seg_img = nc.nCut( 
                preprocessed, cuts=cuts * ((i + 1)) / 2,
                compactness=comp, thresh=thresh * (5 ** i), n_cuts= 6 + i)
                seg_img = np.where(indexes, 0, seg_img)
                mask_ncut = nc.gaussian_mask(seg_img, img_threshold)
                res_img = np.where(indexes, 0, res_img)
                res_img, mask = nc.jointRegions(res_img, seg_img, mask_ncut, fix, 0)
            else: 
                break
            i += 1
    best = min(history, key=lambda x: x['score'])
    return best['result'], best['segment'], best['threshold']

