#%%
"""
GH group data image processing script
"""
#%% load all modules
import LoadDat
import Viewer
"""
Notes on some general functions.

1. Viewer package
    - this package should include all data viewer scripts.
    - Currently available:
        Viewers: providing basic view of any image array or ict dat array
            - launch_image_viewer(image): pop up GUI for image view. image format is (N-shots, X, Y)
            - launch_ict_viewer(ict_dict, sec_per_div=200e-9): pop up GUI for ict profile view.
                Example:
                    ict_dict = {
                        'Ch1_wfm': dat[0]['Ch1_wfm'],
                        'Ch2_wfm': dat[0]['Ch2_wfm'],
                    }
                    launch_ict_viewer(ict_dict, sec_per_div=200e-9)
"""
#%% Data loading - may need update depending on the facility's standard data format
"""
All module must have the following form
    dat = load(filenames = [],datType = [] )
        Loading data and remove duplicate data and do synchronization for image and ict.
    Input
        filenames: list of numpy data file names. If empty, open up windows GUI for selection.
        datType: provides all data that needes to be returned (e.g., 'image', 'ict_ch1', 'Ch2_wfm')
    Output
        dat: list including *.npy data

Currently available:
    AWAnpOut: loading *.npy format data
"""
dat = LoadDat.AWAnpOut.load([],datType=[])
#%% Data check
"""
Check any image you want to check at this point for any purpose
for single instance, for example,
    Viewer.Viewers.launch_image_viewer(dat[i]['image'])
    ict_dict = {
        'Ch1_wfm': dat[0]['Ch1_wfm'],
        'Ch2_wfm': dat[1]['Ch2_wfm'],
    }
    Viewer.Viewers.launch_ict_viewer(ict_dict, sec_per_div=200e-9)
for multi instance,
    extra variable to keep reference should exist, so add empty list like below and use append.
        img_viewers = []
        ict_viewers = []
        img_viewers.append(  Viewer.Viewers.launch_image_viewer(dat[i]['image'])    )
"""
img_viewers = []
ict_viewers = []
N = len(dat)
for i in range(N):
    img_viewers.append(  Viewer.Viewers.launch_image_viewer(dat[i]['image'])    )
    ict_dict = {
        'Ch1_wfm': dat[i]['Ch1_wfm'],
        'Ch2_wfm': dat[i]['Ch2_wfm'],
    }
    ict_viewers.append(   Viewer.Viewers.launch_ict_viewer(ict_dict, sec_per_div=200e-9)    )
#%% 
