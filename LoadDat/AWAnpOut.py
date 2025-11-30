"""
AWA, numpy type data load and data synchronization check

@author: Gwanghui

load(filenames = [],datType = [] )
filenames: list of numpy data file names
datType: provides all data that needes to be returned (e.g., 'image', 'ict_ch1', 'Ch2_wfm')

"""
import os
import numpy as np
from qtpy.QtWidgets import QFileDialog

def load(filenames=None, datType=None):

    # ---------- filenames selection ----------
    if not filenames:
        start_dir = os.getcwd()
        filenames, _ = QFileDialog.getOpenFileNames(
            None,
            "Select numpy data files",
            start_dir,
            "NumPy files (*.npy)",
        )
        filenames = list(filenames)

    if not filenames:
        return []

    for i, f in enumerate(filenames):
        print(f"{i}: {f}")

    # what user requested (None/[] => "all keys")
    if datType is None or datType == []:
        datType_user = None
    else:
        datType_user = list(datType)

    # definitions consistent with validate_dict
    img_key   = 'image'
    img_stamp = 'img_stamp'
    ict_forms = ['ict_ch1','ict_ch2','ict_ch3','ict_ch4',
                 'Ch1_wfm','Ch2_wfm','Ch3_wfm','Ch4_wfm']
    ict_stamp = 'ict_stamp'

    dat = []

    # ---------- loop over files ----------
    for fname in filenames:
        temp = np.load(fname, allow_pickle=True)[()]

        # sanity checks (stamp presence, etc.)
        validate_dict(temp)

        # decide what keys to *operate on* for this file
        if datType_user is None:
            # use ALL keys in this file for operations
            datType_eff = list(temp.keys())
        else:
            datType_eff = list(datType_user)

        # ---------------- DUPLICATES (image) ----------------
        if (img_key in datType_eff) and (img_key in temp):
            img_arr = np.asarray(temp[img_key])
            mask_img = duplicate_mask(img_arr)  # boolean of shape (Nshot,)
            temp[img_key] = img_arr[mask_img]
            if img_stamp in temp:
                temp[img_stamp] = np.asarray(temp[img_stamp])[mask_img]

        # ---------------- DUPLICATES (ICT) ------------------
        ict_keys_eff = [k for k in ict_forms if (k in datType_eff) and (k in temp)]
        if ict_keys_eff:
            ref_key = ict_keys_eff[0]
            ref_arr = np.asarray(temp[ref_key])
            mask_ict = duplicate_mask(ref_arr)

            for k in ict_keys_eff:
                temp[k] = np.asarray(temp[k])[mask_ict]
            if ict_stamp in temp:
                temp[ict_stamp] = np.asarray(temp[ict_stamp])[mask_ict]

        # ---------------- SYNCHRONIZATION -------------------
        has_image = (img_key in datType_eff) and (img_key in temp)
        has_ict   = any((k in datType_eff) and (k in temp) for k in ict_forms)

        if has_image and has_ict:
            # only pass image + ict keys to sync
            datType_sync = [img_key] + [k for k in ict_forms if (k in datType_eff) and (k in temp)]
            temp = sync_image_ict(
                temp,
                datType_sync,
                img_key=img_key,
                img_stamp_key=img_stamp,
                ict_stamp_key=ict_stamp,
                tol=0.3,
            )

        # ---------------- RESHAPE IMAGE ---------------------
        if img_key in temp:
            img_arr = np.asarray(temp[img_key])
            # img_arr shape is (Nshot, Npix)
            n_pix = img_arr.shape[-1]

            if   n_pix == 2304000:
                camPix = (1200, 1920)
            elif n_pix == 2228224:
                camPix = (1088, 2048)
            elif n_pix == 4194304:
                camPix = (2048, 2048)
            else:
                raise ValueError(f"Unknown flat image size {n_pix}")

            temp[img_key] = img_arr.reshape(-1, camPix[0], camPix[1])

        # ---------------- WHAT TO RETURN --------------------
        if datType_user is None:
            # user didn't specify; return full dict (after dup/sync/reshape)
            out = temp
        else:
            wanted = set(datType_user)

            # auto-include stamps
            if (img_key in wanted) and (img_stamp in temp):
                wanted.add(img_stamp)
            if any((k in wanted) for k in ict_forms) and (ict_stamp in temp):
                wanted.add(ict_stamp)

            out = {k: temp[k] for k in wanted if k in temp}

        dat.append(out)

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

#%% save/load json
import json
import Processing

def save_processing_info(filepath="processing_info.json"):
    """
    Save Processing.session_state.processing_info to a JSON file.

    Parameters
    ----------
    filepath : str
        Where to save the JSON.
    """
    info = Processing.session_state.processing_info

    # ensure the parent folder exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"[OK] processing_info saved to {filepath}")

def load_processing_info(filepath="processing_info.json"):
    """
    Load JSON file and overwrite Processing.session_state.processing_info.

    Parameters
    ----------
    filepath : str
        JSON file to load.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Overwrite (not merge)
    Processing.session_state.processing_info = loaded

    print(f"[OK] processing_info loaded from {filepath}")

#%% saving data
def save_dat_npy(dat, filepath="dat_saved.npy"):
    """
    Save the 'dat' variable to a .npy file using allow_pickle=True,
    which preserves nested structures (dicts, lists, arrays).
    """
    np.save(filepath, dat, allow_pickle=True)
    print(f"[OK] dat saved to {filepath}")