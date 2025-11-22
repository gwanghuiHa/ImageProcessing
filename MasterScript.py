#%%
"""
GH group data image processing script
Check out Readme for details
"""
#%% load all modules
import LoadDat
import Viewer
import Processing
from matplotlib import pyplot as plt
#%% Data loading - may need update depending on the facility's standard data format
dat = LoadDat.AWAnpOut.load([],datType=[])
dat_bg = LoadDat.AWAnpOut.load([],datType=[])
#%% Data check
fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="all",cmap="viridis",title=None,max_per_fig=24)
Viewer.Viewers.image_viewer_simple(dat_bg[0]['image'][2:],x_axis=None,y_axis=None,mode="average",cmap="viridis",title=None,max_per_fig=24)

# Viewer.Viewers.image_viewer_profiles_fft(    dat[1]['image'],    x_axis=None,    y_axis=None,    mode="single",     cmap="viridis",    title=None)
#%% Angle correction
angle = 0
is_rot_ok = False
img_to_check = dat[0]['image'][0]

if is_rot_ok==False:  # check image and determine the angle
    temp = Processing.GH_tools.rotate_one_image(img_to_check, angle)
    plt.imshow(temp)
else:                 # rotate all image data
    N = len(dat)
    for i in range(N):
        dat[i]['image'] = Processing.GH_tools.rotate_all_image(dat[i]['image'], angle, overwrite=True)
#%% Calibration
ellipse_info = Processing.GH_tools.ellipse_manual(ax= ax, line_color= "orange", line_style= "--", line_width= 1.5)
cal,fid = Processing.GH_tools.conversion_yag(yag=50e-3,ellipse_info=None,overwrite=True)
#%% background substraction
dat[0]['image'] = Processing.GH_tools.bg_substraction(dat[0]['image'], dat_bg[0]['image'][2:])
fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
#%% ROI setup
fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="overlap",cmap="viridis",title=None)
fig,ax = Viewer.Viewers.image_viewer_simple(dat[0]['image'],x_axis=None,y_axis=None,mode="average",cmap="viridis",title=None)
ellipse_info = Processing.GH_tools.ellipse_manual(ax= ax, line_color= "orange", line_style= "--", line_width= 1.5)

#%% Threshold
rect_info = Processing.GH_tools.rectangle_manual(ax= ax, line_color="orange", line_style="--", line_width=1.5)


