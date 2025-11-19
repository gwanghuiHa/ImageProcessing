"""
AWA, numpy type data load and data synchronization check

@author: Gwanghui

load(filenames = [],datType = [] )
filenames: list of numpy data file names
datType: provides all data that needes to be returned (e.g., 'image', 'ict_ch1', 'Ch2_wfm')

"""
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog

def load(filenames = [],datType = [] ):

    # GUI selection if filenames are not provided
    if filenames==[]:
        start_dir = os.getcwd()
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        filenames, _ = QFileDialog.getOpenFileNames(None,"Select numpy data files",start_dir,"Python Files (*.npy)", options=options)
    for i, f in enumerate(filenames):
        print(f"{i}: {f}")
        
    if datType==[]:
        datType = [k for k in datType]

    dat = []
    # looping over files
    for i in range(len(filenames)):
        temp = np.load(filenames[i],allow_pickle=True)[()]
        
        validate_dict(temp)
        
        # --- remove duplicates on IMAGE side (if present) ---
        if 'image' in temp:
            mask_img = duplicate_mask(np.asarray(temp['image']))
            # apply to image and img_stamp
            temp['image']     = np.asarray(temp['image'])[mask_img]
            temp['img_stamp'] = np.asarray(temp['img_stamp'])[mask_img]

        # --- remove duplicates on ICT side (if present) ---
        ict_keys = [k for k in datType if k != 'image' and k in temp]
        if ict_keys:
            # use the first ict key as reference for duplicates
            ref_key = ict_keys[0]
            mask_ict = duplicate_mask(np.asarray(temp[ref_key]))

            # apply to all ict-related arrays that share ict index
            for k in ict_keys:
                temp[k] = np.asarray(temp[k])[mask_ict]
            temp['ict_stamp'] = np.asarray(temp['ict_stamp'])[mask_ict]

        # image and ict synchronization
        if ('image' in datType) and len(datType)>1:
            temp = sync_image_ict(temp, datType, img_key='image', img_stamp_key='img_stamp', ict_stamp_key='ict_stamp', tol=0.3)
        dat.append(temp)
  
        # --- image 1D array to nominal 2D array ---
        if temp['image'][0].shape[0] == 2304000:
            camPix = (1200, 1920)
        elif temp['image'][0].shape[0] == 2228224:
            camPix = (1088, 2048)
        elif temp['image'][0].shape[0] == 4194304:
            camPix = (2048, 2048)

        temp['image'] = temp['image'].reshape(-1,camPix[0],camPix[1])
    
    return dat

def sync_image_ict(dat, datType, img_key='image', img_stamp_key='img_stamp', ict_stamp_key='ict_stamp', tol=0.3):

    ict_keys = [k for k in datType if k != img_key and k in dat]

    img_stamp = np.asarray(dat[img_stamp_key])
    ict_stamp = np.asarray(dat[ict_stamp_key])

    Ni = len(img_stamp)
    Nj = len(ict_stamp)

    # sort timestamps for efficient matching
    img_order = np.argsort(img_stamp)
    ict_order = np.argsort(ict_stamp)

    i = j = 0
    matched_img_idx = []
    matched_ict_idx = []

    while i < Ni and j < Nj:
        ii = img_order[i]
        jj = ict_order[j]
        ti = img_stamp[ii]
        tj = ict_stamp[jj]
        diff = ti - tj

        if abs(diff) <= tol:
            # match found
            matched_img_idx.append(ii)
            matched_ict_idx.append(jj)
            i += 1
            j += 1
        elif diff < -tol:
            # image earlier than ict, no ict close enough → skip this image
            i += 1
        else:
            # ict earlier than image, no image close enough → skip this ict
            j += 1

    matched_img_idx = np.array(matched_img_idx, dtype=int)
    matched_ict_idx = np.array(matched_ict_idx, dtype=int)

    # If nothing matches, you may want to raise
    if len(matched_img_idx) == 0:
        raise ValueError("No synchronized image–ict pairs found within tolerance.")

    # Rebuild image-side arrays
    dat[img_key] = np.asarray(dat[img_key])[matched_img_idx]
    dat[img_stamp_key] = img_stamp[matched_img_idx]
    
    # --- rebuild ICT side (all ict_keys + ict timestamp) ---
    for k in ict_keys:
        arr = np.asarray(dat[k])
        if arr.shape[0] != Nj:
            raise ValueError(f"Key '{k}' has first dimension {arr.shape[0]}, "
                             f"expected {Nj} to match ict_stamp.")
        dat[k] = arr[matched_ict_idx]

    dat[ict_stamp_key] = ict_stamp[matched_ict_idx]

    return dat


def duplicate_mask(arr):
    n = arr.shape[0]
    keep = [True]  # always keep first

    for i in range(1, n):
        if np.array_equal(arr[i], arr[i-1]):
            keep.append(False)   # duplicate → drop
        else:
            keep.append(True)    # new data → keep

    return np.array(keep, dtype=bool)

def validate_dict(temp):
    img_forms = ['image']
    ict_forms = ['ict_ch1','ict_ch2','ict_ch3','ict_ch4','Ch1_wfm','Ch2_wfm','Ch3_wfm','Ch4_wfm']
    img_stamp = 'img_stamp'
    ict_stamp = 'ict_stamp'
    
    # check if ANY image form exists
    has_img = any(k in temp for k in img_forms)

    # check if ANY ict form exists
    has_ict = any(k in temp for k in ict_forms)

    # check presence of timestamp keys
    has_img_stamp = img_stamp in temp
    has_ict_stamp = ict_stamp in temp

    # validation logic
    if has_img and not has_img_stamp:
        raise ValueError("Image data present but NO image timestamp found.")

    if has_ict and not has_ict_stamp:
        raise ValueError("ICT data present but NO ICT timestamp found.")

    return True


