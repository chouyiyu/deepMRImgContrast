# deep MR Image Contrast Classification
automatically identify the contrast of brain MRI scans using deep learning 
## Installation
```
git clone https://github.com/chouyiyu/deepMRImgContrast.git
```
## Prerequisites
```
python3
keras
tensorflow
numpy
nibabel
urllib
argparse
```
## How to use it
option 1: classify the MR image contrast by finding the shortest distance among the support set saved under /template including T1, post contrast T1, T2, FLAIR and post contrast FLAIR sequences.
```
python3 deepImgContrast.py --mode classify --img1 /path/to/image --gpu 0
```
option 2: compute the distance (similarity) between image1 and image2 in the embedding space
```
python3 deepImgContrast.py --mode dist --img1 /path/to/image1 --img2 /path/to/image2 --gpu 0 
```
Input images must be in NIfTI format and rigidly registered to the MNI space (template provided as MNI_template.nii) with dimension 92x108x92 and voxel size 2mm^3. By default, deepImgContrast will run on CPU mode, set the option --gpu to 0,1,2 .... for running on GPU mode. 
#

This project is licensed under the terms of the GNU General Public License v3.0
