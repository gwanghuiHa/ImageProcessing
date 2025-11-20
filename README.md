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