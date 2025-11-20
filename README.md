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
│
```
## Documentation
### LoadDat
This module includes File I/O functions.

#### AWAnpOut: loading *.npy format data  
>  **Usage**  
>    dat = LoadDat.AWAnpOut.load([],datType=[])  
>  **Input**  
>    filenames: list of numpy data file names. If empty, open up windows GUI for selection.  
>    datType: keys for the data list (e.g., 'image', 'ict_ch1', 'Ch2_wfm'). If empty, all keys will be considered.  
>  **Output**  
>    dat: list including *.npy data  
