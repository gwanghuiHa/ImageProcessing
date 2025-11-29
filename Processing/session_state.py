#%%
"""
Master dictionary shared across whole processing modules
"""

processing_info = {
    "rotation_angle": 0,
    
    "ellipse_info": None,
    "rect_info": None,
    "lasso_info":None,
    
    "calibration": None,
    "fiducial": None,
    
    "Calibration_ellipse": None,
    "Calibration_rect": None,
    
    "ROI_ellipse": None,
    "ROI_rect": None,
    "ROI_lasso": None,

    "Threshold_ellipse": None,
    "Threshold_rect": None,
    "Threshold_lasso": None,
    
    "median_window": 3
    }