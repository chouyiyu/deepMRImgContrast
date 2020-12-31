# deep MR Image Contrast Classification
automatically identify the contrast of brain MRI scans using deep learning 
## Installation
```

```
## How to use it
```
python3 deepImgContrast.py --img1 /path/to/image1 --img2 /path/to/image2 --gpu 0 
```
Input images must be in NIfTI format and rigidly registered to the MNI152 template and then downsampled to 2mm isotropic. By default, deepImgContrast will run in CPU mode, set the option --gpu to 0,1,2.... for running in GPU mode. The output is the distance/similarity between img1 and img2 measured in the embedding space.
T1, post contrast T1, T2, FLAIR and post contrast FLAIR templates were provided (saved under /template), serving as a support set for the classification of MR image constrast by finding the shortest distance among the pairs.
