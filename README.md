# Vessel Segmentation
This repository contains codes for vessel segmentation for MRA images and can be applied to other 3D datasets.


## Dataset: https://public.kitware.com/Wiki/TubeTK/Data

## Preprocessing

Proprocessing is done using the repository: https://github.com/quqixun/BrainPrep
![Vessel_Segmentation_Preprocessing](./figures/preprocessing.png)

## Patch Generation
First create the csv files contains dataset list of images and labels.
```
python generate_dataframe_path.py
```
And then you can create the 3D patch of your desired size 

```
python generate_3D_patches.py

```
