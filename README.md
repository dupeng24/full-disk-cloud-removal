# Cloud Removal in Full-disk Solar Images Using Deep Learning
This project leverages deep learning for cloud removal in full-disk solar images, encompassing both datasets and network structure. 
## Setup
Project Clone to Local
```
git clone https://github.com/dupeng24/cloud-removal.git
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
Please run the following to generate the h5 dataset file:
```
python makedataset.py
```
To train the models in the paper, run these commands:
```
python train.py
```
## Testing
To conduct testing, save the trained model weights in the checkpoint folder and rename them accordingly,run these commands:
```
python test.py
```
## Acknowledgement
The authors express their gratitude to the Global Oscillation Network Group (GONG) and the Huairou Solar Observing Station (HSOS) for providing the data. 
