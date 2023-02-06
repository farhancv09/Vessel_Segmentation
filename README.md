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
And then you can create the 3D patch of your desired size. 

```
python generate_3D_patches.py

```

## Training

There are other architetures as well in architecture/ folder which were most commonly used for segmentation. 
```
python Training.py
```
![Proposed_architecture](./figures/Architecture.png)

## Inference
Inference is done using sliding window method.

```
python Inference.py

```
# Results
## Quantitaive
![Comparison Table](./figures/Comparison.png)
## Quantitative
![Comparison Table](./figures/Comparison_table.png)

