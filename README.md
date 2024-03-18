# genetic-conditions-image-classifier
  This repository contains official implementation of classifer for "Approximating facial expression effects on diagnostic accuracy via generative AI in medical genetics". 
  We have used the code from official paper 
  > Ömer Sümer, Rebekah L. Waikel, Suzanna E. Ledgister Hanchard, Dat Duong, Cristina Conati, Peter Krawitz, Benjamin D. Solomon, and Elisabeth André, "Region-based Saliency Explanations on the Recognition of Facial Genetic Syndromes," Machine Learning for Healthcare, 2023.

You can access the metadata files to reproduce the database used in this study:
NIH Facial Genetic Syndromes Database [(Zenodo dataset link)]([http://doi.org/10.5281/zenodo.8113907](https://doi.org/10.5281/zenodo.8113906))
## Environment

All required Python packages are listed in the ```ènvironment.yml````file. Create and activate Python environment

```
conda env create --name facial-gestalt-xai --file environment.yml
conda activate facial-gestalt-xai
``` 
Tested in Ubuntu 22.04 OS using Python 3.9.16 in a Conda environment.

## Training Baseline DNN

We used VGG Face-2 pretrained ResNet weights for initialization and ResNet-50 architecture [(link)](https://github.com/cydonia999/VGGFace2-pytorch)

Reference paper: ZQ. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman, VGGFace2: A dataset for recognising faces across pose and age, 2018.

* Download VGGFace2 pretrained ResNet50 weights to the following directory:
* ```/models/weights/resnet50_ft_weight.pkl```
   **Repository:** https://github.com/cydonia999/VGGFace2-pytorch \
   **G-Drive download link:** https://drive.google.com/file/d/1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU/view

* Train baseline model on recognizing facial genetic syndromes (NIH-Faces dataset)
   ```
   python train.py --dataset_folder $DATASET_ROOT --fold fold-1
   ```

* After training all 5 folds, evaluate accuracy and F1-scores of these models:
   ```
   python test.py --dataset_folder $DATASET_ROOT
