# Code for "Steerable Pyramid Transform Enables Robust Left Ventricle Quantification"

This is an end-to-end framework for accurate and robust left ventricle indices quantification, including cavity and myocardium areas, six regional wall thicknesses, and three directional dimensions. 

The proposed method first decomposes a CMR image into directional frequency bands via Steerable Pyramid Transformation. Then the deep representation of each direction is extracted separately via a CNN model, and we also use an LSTM module to model the temporal dynamics. Finally, we explore the multidirectional relationship of features, indices, and directional subbands to optimize the quantification system. 

<img src="https://github.com/yangyangyang127/LVquant/blob/master/wholeframework.png" width="800" >

### Requirements
Create a conda environment and install dependencies:
```bash
cd RobustLV

conda create -n RobustLV python=3.7
conda activate RobustLV

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit

pip install -r requirements.txt
pip install pointnet2_ops_lib/.
```
If you want to test the Mamba module, please refer to [VMamba](https://github.com/MzeroMiko/VMamba/tree/main) to set up the environment.


### Datasets

The dataset we used can be found at [the MICCAI 2018/2019 Left Ventricle Full Quantification Challenge](https://lvquan19.github.io/), an open-source dataset on Kaggle. The dataset can be put under the './data/' path.


### Training
Train the model with the below command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

Please modify the corresponding hyperparameters to conduct experiments in 'config.py' file. 



## Acknowledgement
We thank [VMamba](https://github.com/MzeroMiko/VMamba/tree/main) and [MTLearn](https://github.com/thuml/MTlearn) for sharing their source code.





