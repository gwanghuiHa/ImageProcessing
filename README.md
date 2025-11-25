# ImageProcessing
Image processing script for GH group

## Structure
```
Mainscript
│
├──LoadDat
│  └──AWAnpOut
│      └──load(filenames=[], datType=[])
│
├──Viewer
│  └──Viewers
│     ├──image_viewer_simple(images,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
│     └──ict_viewer(ict_wfm, ns_per_div=200, mode="single", title=None)
│
├──Processing
│  └──GH_tools
│     ├──rotate_one_image(img, angle_deg)
│     ├──rotate_all_image(img, angle_deg)
│     ├──ellipse_manual(ax, line_color="orange", line_style="--", line_width=1.5,)
│     ├──rectangle_manual(ax, line_color="orange", line_style="--", line_width=1.5,)
│     ├──conversion_yag(yag=50e-3,ellipse_info=None,overwrite=True)
│     ├──bg_substraction(img_main, img_bg)
│     ├──apply_roi_mask(image, roi_info=None, roi_type="ellipse", outside_value=0)
│     ├──apply_roi_threshold(images, roi_info=None, roi_type="ellipse", scaling=1.0, save_scaling=True)
│     ├──apply_median_filter(images, window_size=3, save=True)
│     ├──
│     ├──
│     ├──
│     ├──
```
## Documentation
### LoadDat
This module includes File I/O functions.

#### AWAnpOut: loading *.npy format data - GH
>  **dat = LoadDat.AWAnpOut.load([],datType=[])**  
>  **Input**  
>  &nbsp;&nbsp;&nbsp;&nbsp;filenames: list of numpy data file names. If empty, open up windows GUI for selection.  
>  &nbsp;&nbsp;&nbsp;&nbsp;datType: keys for the data list (e.g., 'image', 'ict_ch1', 'Ch2_wfm'). If empty, all keys will be considered.  
>  **Output**  
>  &nbsp;&nbsp;&nbsp;&nbsp;dat: list including *.npy data  

### Viewer
This module includes all image or ict wave form viewers  

#### Viewers: image and ict data viewers - GH  
>  **image_viewer_simple(images,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)**  
>  **Input**  
>  &nbsp;&nbsp;&nbsp;&nbsp;images: numpy array with the dimension of (N-shots, X, Y)  
>  &nbsp;&nbsp;&nbsp;&nbsp;x(y)_axis: x- and y-axis array. If None, just use pixel numbers  
>  &nbsp;&nbsp;&nbsp;&nbsp;mode:  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - single: shows one image. You can still load entire image and use slider to see other shots.  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - all: shows entire shots using small subplots  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - overlap: shows one added up image  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - average: shows one averaged image    
>  
>  **Output**  
>  &nbsp;&nbsp;&nbsp;&nbsp;fig,ax: figure and axis info for later control  

>  **ict_viewer(ict_wfm, ns_per_div=200, mode="single", title=None)**  
>  **Input**  
>  &nbsp;&nbsp;&nbsp;&nbsp;ict_wfm: list including ict waveforms for each shot  
>  &nbsp;&nbsp;&nbsp;&nbsp;ns_per_div: scope scale  
>  &nbsp;&nbsp;&nbsp;&nbsp;mode:  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - single: shows one wave form. You can still load entire waveforms and use slider to see other shots.  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - all: shows entire shots using small subplots  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - overlap: shows one added up waveform  
>  &nbsp;&nbsp;&nbsp;&nbsp;  - average: shows one averaged waveform   
>
>  **Output**  
>  &nbsp;&nbsp;&nbsp;&nbsp;fig,ax: figure and axis info for later control  

### Processing
This module includes various tools to process the image or ict signal  

#### GH_tools: processing tools that GH usually uses - GH
> **rotated = rotate_one_image(img, angle_deg=None)**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotate single image  
> **Input**  
>   &nbsp;&nbsp;&nbsp;&nbsp;img: original image array.  
>   &nbsp;&nbsp;&nbsp;&nbsp;angle_deg: rotation angle in deg. if this is None or [], it load data from session_state.  
> **Output**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotated: rotated image.  

> **rotated = rotate_all_image(img, angle_deg=None, overwrite=True)**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotate all provided images.  
> **Input**  
>   &nbsp;&nbsp;&nbsp;&nbsp;img: original image array including N-shots (Nshot, X,Y).  
>   &nbsp;&nbsp;&nbsp;&nbsp;angle_deg: rotation angle in deg. if this is None or [], it load data from session_state.  
>   &nbsp;&nbsp;&nbsp;&nbsp;overwrite: if this is true, angle used in this function will be uploaded to the session_state.  
> **Output**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotated: rotated images.  

> **ellipse_info = ellipse_manual(ax, line_color="orange", line_style="--", line_width=1.5,)**    
> &nbsp;&nbsp;&nbsp;&nbsp;manually drawing ellipse to on the canvas. You must open any type of canvas and provide its axes info.  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp;ax: pyplot figure's axis info.  
> **Output**  
> &nbsp;&nbsp;&nbsp;&nbsp;ellipse_info: dictionary including center_x, center_y, width, and height of ellipse.  

> **rect_info = rectangle_manual(ax= ax, line_color="orange", line_style="--", line_width=1.5)**  
> &nbsp;&nbsp;&nbsp;&nbsp;   manually drawing rectangle to on the canvas. You must open any type of canvas and provide its axes info.  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp;    ax: pyplot figure's axis info.    
> **Output**  
> &nbsp;&nbsp;&nbsp;&nbsp;    rect_info: dictionary including center_x, center_y, width, and height of rectangle.  

> **cal,fiducial = conversion_yag(yag=50e-3,ellipse_info=None,overwrite=True)**  
> &nbsp;&nbsp;&nbsp;&nbsp;   using ellipse info update calibration factor and fiducial  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp;    yag: size of the yag in meter  
> &nbsp;&nbsp;&nbsp;&nbsp;    ellipse_info: ellipse dictionary. if it is None, the fuction will try to get ellipse_info from session_state.  
> &nbsp;&nbsp;&nbsp;&nbsp;    overwrite: if this is False, then the function will try to get info from sesson_state. Otherwise, it update the session_state.  

> **img = bg_substraction(img_main, img_bg)**  
> &nbsp;&nbsp;&nbsp;&nbsp;    average background image and substract it from main images  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp;    img_main: main images in the form of (Nshot, X, Y)  
> &nbsp;&nbsp;&nbsp;&nbsp;    img_bg: background images in the form of (Nshot, X, Y)  
> **Output**  
> &nbsp;&nbsp;&nbsp;&nbsp; substracted main image array  

> **out = apply_roi_mask(image, roi_info=None, roi_type="ellipse", outside_value=0)**  
> &nbsp;&nbsp;&nbsp;&nbsp;    Zero all pixels outside of ROI.  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp;    images: array, 2D (H, W) or 3D (N, H, W) image(s).  
> &nbsp;&nbsp;&nbsp;&nbsp;    roi_info: dict or [] or None  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        If dict, must be:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            {'center_x', 'center_y', 'width', 'height'} in pixel coordinates.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        If [] or None, ROI will be loaded from  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            session_state.processing_info["ellipse_info"]  (for roi_type='ellipse')  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            session_state.processing_info["rect_info"]     (for roi_type='rectangle')  
> &nbsp;&nbsp;&nbsp;&nbsp;    roi_type: {'ellipse','rect'}  
> &nbsp;&nbsp;&nbsp;&nbsp;    outside_value: scalar value to assign outside the ROI (default 0).  
> **Output**  
> &nbsp;&nbsp;&nbsp;&nbsp;    out: roi applied images=(s)  

> **out = apply_roi_threshold(images, roi_info=None, roi_type="ellipse", scaling=1.0, save_scaling=True)**  
> &nbsp;&nbsp;&nbsp;&nbsp; ROI-based background thresholding.  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp; images : array, 2D (H, W) or 3D (N, H, W). If 3D, mean over shots is used.   
> &nbsp;&nbsp;&nbsp;&nbsp; roi_info : dict or [] or None  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    If dict, must be:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      {'center_x', 'center_y', 'width', 'height'} in pixels.   
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    If [] or None, ROI will be loaded from  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      session_state.processing_info["ellipse_info"]  (roi_type='ellipse')  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      session_state.processing_info["rect_info"]     (roi_type='rectangle')  
> &nbsp;&nbsp;&nbsp;&nbsp;roi_type : {'ellipse','rect'}  
> **Output**  
> &nbsp;&nbsp;&nbsp;&nbsp; out : Thresholded image(s), same shape and dtype as input.

> **out = apply_median_filter(images, window_size=3, save=True)**  
> &nbsp;&nbsp;&nbsp;&nbsp;    Apply median filter to 2D or 3D images.  
> &nbsp;&nbsp;&nbsp;&nbsp;    Saves the window size to session_state.processing_info[save_key].  
> **Input**  
> &nbsp;&nbsp;&nbsp;&nbsp; images : ndarray (H,W) or (N,H,W)  
> &nbsp;&nbsp;&nbsp;&nbsp;    window_size : int,   Median filter window size (must be odd).  
> &nbsp;&nbsp;&nbsp;&nbsp;    save : bool  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        If True, save window_size into session_state.processing_info.  
> **Output**  
> &nbsp;&nbsp;&nbsp;&nbsp;    out : ndarray, Same shape as input, with median filtering applied.  
