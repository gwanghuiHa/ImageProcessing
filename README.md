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
│     └──rotate_all_image(img, angle_deg)

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
> **rotated = rotate_one_image(img, angle_deg)**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotate single image  
> **Input**  
>   &nbsp;&nbsp;&nbsp;&nbsp;img: original image array.  
>   &nbsp;&nbsp;&nbsp;&nbsp;angle_deg: rotation angle in deg.  
> **Output**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotated: rotated image.  

> **rotated = rotate_all_image(img, angle_deg)**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotate all provided images.  
> **Input**  
>   &nbsp;&nbsp;&nbsp;&nbsp;img: original image array including N-shots (Nshot, X,Y).  
>   &nbsp;&nbsp;&nbsp;&nbsp;angle_deg: rotation angle in deg.  
> **Output**  
>   &nbsp;&nbsp;&nbsp;&nbsp;rotated: rotated images.  

