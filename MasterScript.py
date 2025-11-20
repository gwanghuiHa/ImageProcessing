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
#%% Data check
fig,ax = Viewer.Viewers.image_viewer_simple(dat[1]['image'],x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
Viewer.Viewers.image_viewer_simple(dat[1]['image'],x_axis=None,y_axis=None,mode="all",cmap="viridis",title=None,max_per_fig=24)

Viewer.Viewers.ict_viewer(dat[1]['Ch1_wfm'], ns_per_div=200, mode="overlap", title=None)

#%% Angle correction
"""
Processing.GH_tools.rotate_one_image(img, angle_deg)
    it rotates one image
    img: original image array
    angle_deg: rotation angle in deg. + counter-clock.
Processing.GH_tools.rotate_all_image(img, angle_deg)
    it rotates all images. img should be (Nshot, X, Y)
"""
angle = 45
is_rot_ok = False
img_to_check = dat[0]['image'][0]

if is_rot_ok==False:  # check image and determine the angle
    temp = Processing.GH_tools.rotate_one_image(img_to_check, angle)
    plt.imshow(temp)
else:                 # rotate all image data
    N = len(dat)
    for i in range(N):
        dat[i]['image'] = Processing.GH_tools.rotate_all_image(dat[i]['image'], angle)
#%% Calibration
"""
Processing.GH_tools.rotate_one_image(img, angle_deg)
    it rotates one image
    img: original image array
    angle_deg: rotation angle in deg. + counter-clock.
Processing.GH_tools.rotate_all_image(img, angle_deg)
    it rotates all images. img should be (Nshot, X, Y)
"""


