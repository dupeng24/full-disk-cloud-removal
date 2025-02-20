# Cloud Removal in Full-disk Solar Images Using Deep Learning
This project leverages deep learning for cloud removal in full-disk solar images, encompassing both datasets, model parameters and network structure. 

**Title of the Article**: [Cloud Removal in Full-disk Solar Images Using Deep Learning](https://iopscience.iop.org/article/10.3847/1538-4365/ad93ca)  
**DOI**: [10.3847/1538-4365/ad93ca](https://doi.org/10.3847/1538-4365/ad93ca)  

![5](https://github.com/user-attachments/assets/df9370c6-4862-4654-b445-f7fa0be70be8)
## Setup
Project Clone to Local
```
git clone https://github.com/dupeng24/full-disk-cloud-removal.git
```
To install dependencies:

```
conda install python=3.11.4
conda install pytorch-cuda=11.8
conda install numpy=1.25.0
conda install scikit-image=0.20.0
conda install h5py=3.9.0
```
## Cloud Detection and Classification
To conduct cloud detection and categorization, please change the name of the corresponding folder first, run these commands:
```
python finallyquality.py
```
## Training
Please download the data compression package full-disk images data.zip, divide the training set in it into cloud and clean, please make sure that the cloudy image and the labeled image have the same name. Please run the following to generate the h5 dataset file:
```
python makedataset.py
```
To train the models in the paper, run these commands:
```
python train.py
```
## Testing
To conduct testing, please download the model parameter zip without unzipping it and place it directly under a checkpoint folder. Set the input and output paths, run these commands:
```
python test.py
```
## Examples of experimental results
input

<img src="https://github.com/user-attachments/assets/22511be3-1b47-4099-af50-812a32c53b2e" width="200px">
<img src="https://github.com/user-attachments/assets/98ee5493-2d7f-4a8a-80ef-01f3cca7b79f" width="200px">
<img src="https://github.com/user-attachments/assets/14869ec5-b442-4b62-b895-3c143a40e616" width="200px">

output

<img src="https://github.com/user-attachments/assets/58aa78bb-9288-43d0-be77-d0395ec79e58" width="200px">
<img src="https://github.com/user-attachments/assets/e303c24c-d487-43bb-ba22-a3230925598b" width="200px">
<img src="https://github.com/user-attachments/assets/39003091-7849-42f9-86aa-3fc0dbb9b2ae" width="200px">

## Acknowledgement
Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA. 
The authors express their gratitude to the Global Oscillation Network Group (GONG) for providing the data. 
