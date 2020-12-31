# deep MR Image Contrast Classification
automatically identify the contrast of brain MRI scans using deep learning 
## Installation
```
git clone https://github.com/chouyiyu/deepMRImgContrast.git
```
## How to use it
1) compute the similarity (distance) between image1 and image2 measured in the embedding space
```
python3 deepImgContrast.py --mode dist --img1 /path/to/image1 --img2 /path/to/image2 --gpu 0 
```
2) classify the MR image contrast by finding the shortest distance among the support set saved under /template including T1, post contrast T1, T2, FLAIR and post contrast FLAIR sequences.
```
python3 deepImgContrast.py --mode classify --img1 /path/to/image --gpu 0
```
Input images must be in NIfTI format, rigidly registered to the MNI152 template and downsampled to 2mm isotropic. By default, deepImgContrast will run in CPU mode, set the option --gpu to 0,1,2.... for running in GPU mode. 
