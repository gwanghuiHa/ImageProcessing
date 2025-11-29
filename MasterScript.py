#%%
"""
GH group data image processing script
Check out Readme for details

Following shows exampled process.
Multiple switches are added so that users can simply try it and follow the process. But, those are not necessary.
"""
#%% load all modules
import LoadDat
import Viewer
import Processing
from matplotlib import pyplot as plt
import numpy as np
#%% Data loading - may need update depending on the facility's standard data format
dat = LoadDat.AWAnpOut.load([],datType=[])
dat_bg = LoadDat.AWAnpOut.load([],datType=[])

#%% Data check - Image
check_individual = True
check_all_at_once = True
check_averaged = True
check_overlapped = True

if check_individual == True:
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
    fig2,ax2 = Viewer.Viewers.image_viewer_simple(dat_bg[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
if check_all_at_once == True:
    Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="all",cmap="viridis",title=None,max_per_fig=24)
    Viewer.Viewers.image_viewer_simple(dat_bg[0]['image'],x_axis=None,y_axis=None,mode="all",cmap="viridis",title=None,max_per_fig=24)
if check_averaged == True:
    Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="average",cmap="viridis",title=None,max_per_fig=24)
    Viewer.Viewers.image_viewer_simple(dat_bg[0]['image'][2:],x_axis=None,y_axis=None,mode="average",cmap="viridis",title=None,max_per_fig=24)
if check_overlapped == True:
    Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="overlap",cmap="viridis",title=None,max_per_fig=24)
    Viewer.Viewers.image_viewer_simple(dat_bg[0]['image'][2:],x_axis=None,y_axis=None,mode="overlap",cmap="viridis",title=None,max_per_fig=24)

#%% Angle correction
angle = 0
is_rot_ok = True
img_to_check = dat[0]['image'][0]

if is_rot_ok==False:  # check image and determine the angle
    temp = Processing.GH_tools.rotate_one_image(img_to_check, angle)
    plt.imshow(temp)
else:                 # rotate all image data
    N = len(dat)
    for i in range(N):
        dat[i]['image'] = Processing.GH_tools.rotate_all_image(dat[i]['image'], angle, overwrite=True)

#%% Calibration
#calibration_types = ['yag','usaf'] # doesn't support usaf now.
cal_type = 'yag'
ref_length = 50e-3

fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
if cal_type == 'yag':
    ellipse_info = Processing.GH_tools.ellipse_manual(ax= ax, line_color= "orange", line_style= "--", line_width= 1.5)
    Processing.session_state.processing_info['Calibration_ellipse'] = ellipse_info
    cal,fid = Processing.GH_tools.conversion_yag(yag=ref_length,ellipse_info=None,overwrite=True)

#%% Processing
# flow 1: background substraction >> applying ROI >> applying threshold >> median fiter
# flow 2: background substraction >> applying ROI >> morphological clean with threshold
# flow 3: excluding YAG edge >> background substraction (scale) >> morphological clean with threshold 
# flow 4: excluding YAG edge >> applying threshold >> morphological clean >> background substraction (scale)

selection_switch = 'flow3'

if selection_switch == 'flow1':
    fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

    # background substraction
    img = Processing.GH_tools.bg_substraction(dat[0]['image'], dat_bg[0]['image'])

    # applying ROI - lasso, ellipse, rectangle
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="overlap",cmap="viridis",title=None)
    lasso_info = Processing.GH_tools.lasso_manual( ax1, line_color="orange", line_style="--", line_width=1.5)
    Processing.session_state.processing_info['ROI_lasso'] = lasso_info
    img = Processing.GH_tools.apply_roi_mask(img, roi_info=None, roi_type="lasso", outside_value=0)

    # applying background noise removal (threshold)
    plt.close(fig1)
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="overlap",cmap="viridis",title=None)
    rect_info = Processing.GH_tools.rectangle_manual(ax= ax1, line_color= "orange", line_style= "--", line_width= 1.5)
    Processing.session_state.processing_info['Threshold_rect'] = rect_info
    img = Processing.GH_tools.apply_roi_threshold(img, roi_info=None, roi_type="rect", scaling=1.0, save_scaling=True)
    
    # applying median filter
    img = Processing.GH_tools.apply_median_filter(img, window_size=5, save=True)
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

elif selection_switch == 'flow2':
    fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

    # background substraction
    img = Processing.GH_tools.bg_substraction(dat[0]['image'], dat_bg[0]['image'])

    # applying ROI - lasso, ellipse, rectangle
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="overlap",cmap="viridis",title=None)
    ellipse_info = Processing.GH_tools.ellipse_manual( ax1, line_color="orange", line_style="--", line_width=1.5)
    Processing.session_state.processing_info['ROI_ellipse'] = ellipse_info

    img = Processing.GH_tools.apply_roi_mask(img, roi_info=None, roi_type="lasso", outside_value=0)
    
    # applying median filter
    img = Processing.GH_tools.clean_beam_array(img, thr_ratio=0.01, min_size=5000, do_open=True, do_close=True) 
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

elif selection_switch == 'flow3':
    fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

    # excluding YAG using ROI feature
    ellipse_info = Processing.GH_tools.ellipse_manual(ax= ax, line_color= "orange", line_style= "--", line_width= 1.5)
    Processing.session_state.processing_info['ROI_ellipse'] = ellipse_info
    img = Processing.GH_tools.apply_roi_mask(dat[0]['image'], roi_info=ellipse_info, roi_type="ellipse", outside_value=0)
    img_bg = Processing.GH_tools.apply_roi_mask(dat_bg[0]['image'], roi_info=ellipse_info, roi_type="ellipse", outside_value=0)

    # applying background substraction with scaling feature. You need to select beam area to exclude it from scaling
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
    bg_template = Processing.GH_tools.build_bg_template(img_bg, method="median")
    H, W = bg_template.shape
    roi_mask = np.ones((H,W), dtype=bool)
    rect_info = Processing.GH_tools.rectangle_manual(ax= ax1, line_color= "orange", line_style= "--", line_width= 1.5)
    roi_mask[int(rect_info['center_y']-rect_info['height']/2):int(rect_info['center_y']+rect_info['height']/2), int(rect_info['center_x']-rect_info['width']/2):int(rect_info['center_x']+rect_info['width']/2)] = False
    img,ablist = Processing.GH_tools.subtract_bg_scaled(img, bg_template, roi_mask)
    
    # applying morphological cleaning
    img = Processing.GH_tools.clean_beam_array(img, thr_ratio=0.005, min_size=5000, do_open=True, do_close=True) 
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

elif selection_switch == 'flow4':
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

    # excluding YAG using ROI feature
    ellipse_info = Processing.GH_tools.ellipse_manual(ax= ax1, line_color= "orange", line_style= "--", line_width= 1.5)
    Processing.session_state.processing_info['ROI_ellipse'] = ellipse_info
    img = Processing.GH_tools.apply_roi_mask(dat[0]['image'], roi_info=ellipse_info, roi_type="ellipse", outside_value=0)
    img_bg = Processing.GH_tools.apply_roi_mask(dat_bg[0]['image'], roi_info=ellipse_info, roi_type="ellipse", outside_value=0)

    # background noise removal for both main and background images
    rect_info = Processing.GH_tools.rectangle_manual(ax= ax1, line_color= "orange", line_style= "--", line_width= 1.5)
    Processing.session_state.processing_info['Threshold_rect'] = rect_info
    img = Processing.GH_tools.apply_roi_threshold(img, roi_info=None, roi_type="rect", scaling=1.0, save_scaling=True)
    img_bg = Processing.GH_tools.apply_roi_threshold(img_bg, roi_info=None, roi_type="rect", scaling=1.0, save_scaling=True)
    
    # mophological cleaning for both main and background images
    img = Processing.GH_tools.clean_beam_array(img, thr_ratio=0.0, min_size=5000, do_open=True, do_close=True)    
    img_bg = Processing.GH_tools.clean_beam_array(img_bg, thr_ratio=0.0, min_size=5000, do_open=True, do_close=True)    
    
    fig1,ax1 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

    # background substraction
    # bg_template = Processing.GH_tools.build_bg_template(img_bg, method="mean")
    bg_template = Processing.GH_tools.build_bg_template(img_bg, method="median")
    H, W = bg_template.shape
    roi_mask = np.ones((H,W), dtype=bool)
    rect_info = Processing.GH_tools.rectangle_manual(ax= ax1, line_color= "orange", line_style= "--", line_width= 1.5)
    roi_mask[int(rect_info['center_y']-rect_info['height']/2):int(rect_info['center_y']+rect_info['height']/2), int(rect_info['center_x']-rect_info['width']/2):int(rect_info['center_x']+rect_info['width']/2)] = False
    img,ablist = Processing.GH_tools.subtract_bg_scaled(img, bg_template, roi_mask)

    fig2,ax2 = Viewer.Viewers.image_viewer_simple(img,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)

#%% Data check - ICT
#%% Signal range
#%% Signal background
#%% Signal smoothing
#%% Charge calculation
#%% Saving processing setup

#%% Saving data & windowing

