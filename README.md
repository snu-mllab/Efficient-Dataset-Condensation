# Efficient-Dataset-Condensation
Official PyTorch implementation of **"Dataset Condensation via Efficient Synthetic-Data Parameterization", ICML'22**

![image samples](images/title.png)

> **Abstract** *The great success of machine learning with massive amounts of data comes at a price of huge computation costs and storage for training and tuning. 
Recent studies on dataset condensation attempt to reduce the dependence on such massive data by synthesizing a compact training dataset. 
However, the existing approaches have fundamental limitations in optimization due to the limited representability of synthetic datasets without considering any data regularity characteristics.
To this end, we propose a novel condensation framework that generates multiple synthetic data with a limited storage budget via efficient parameterization considering data regularity. 
We further analyze the shortcomings of the existing gradient matching-based condensation methods and develop an effective optimization technique for improving the condensation of training data information. 
We propose a unified algorithm that drastically improves the quality of condensed data against the current state-of-the-art on CIFAR-10, ImageNet, and Speech Commands.*

## Requirements
- The code has been tested with PyTorch 1.11.0.   
- To run the codes, install efficientnet package ```pip install efficientnet_pytorch```

## Test Condensed Data
#### Download data
- You can download condensed data evaluated in our paper from [Here](https://drive.google.com/drive/folders/1yh0Hf2ia4b-1edMiAr1kXCH4eUcYNfmz?usp=sharing).
- The possible datasets are CIFAR-10, MNIST, SVHN, FashionMNIST, and ImageNet (10, 100 subclasses).
- To test data, download the entire dataset folder (e.g., cifar10) and locate the folder at ```./results```. 

#### Test
- Set ```--data_dir``` and ```--imagenet_dir``` in argument.py to point the folder containing the original dataset. 
- 

## Optimize Condensed Data
