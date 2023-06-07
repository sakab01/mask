import cv2
import numpy as np
from operator import itemgetter
import exposure
import norm_cuts as nc
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
    cv2.imshow("prova", retinex_img)
    img_threshold = threshold(retinex_img)
    mask_ncut = nc.gaussian_mask(ncut, img_threshold)
    sclera_ncut, mask_ncut = nc.jointRegions(img, ncut, mask_ncut, fix_range, 0)
    #sclera_ncut, ncut, img_threshold = improve_precision_ncut(original=img, 
    #img_threshold=img_threshold, seg_img=ncut, mask=mask_ncut, preprocessed=preprocessed, 
    #ret_img=retinex_img, res_img=sclera_ncut, blur_scale=blur_scale, 
    #cuts=imp_cuts, thresh=imp_thresh, comp=imp_comp, fix=imp_fix, gamma=gamma)

    return img_threshold, kmeans, sclera_ncut, ncut

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
