# deepMRImgContrast
automatically identify the contrast of MRI scans
## How to use it
```
python deepImgContrast.py --img1 /path/to/image1 --img2 /path/to/image2 --gpu 0 
```
Input images must be in NIfTI format and rigidly registered to the MNI152 template and then resampled to 2mm isotropic. By default, deepImgContrast will run in CPU mode, set the option --gpu to 0,1,2.... for running in GPU mode. The output is the distance/similarity between img1 and img2 measured in the embedding space.
