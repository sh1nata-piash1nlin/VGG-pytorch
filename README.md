CURRENTLY IMPROVING 
# VGG-pytorch
VGG for Vietnamese Currency recognition

# Abstract
Given that money plays an essential role in everyday life and business transactions, currency recognition becomes crucial, particularly for those who are blind or have visual impairments. To address this issue, we propose a work to help visually challenged people identify different denominations of Vietnamese currency using deep learning techniques. By utilizing deep learning approaches, the system will enable these individuals to recognize banknotes more easily.

# Methodology
Currency recognition has been a focus of research in recent years. There are various ways to identify currency based on the features used for classification and the deep learning model applied. Every currency has specific features that make it easy to classify. In the case of Vietnamese currency, certain key features are essential for accurate recognition. For instance, every Vietnamese banknote has a special value that printed in the surface, which corresponds to the label I will define below.  </br>
Here is my pytorch implementation of the model described in the [VGG paper](https://arxiv.org/pdf/1409.1556) for **Vietnamese Currency recognition**.

# Requirments: 
Here are some basic requirements needed for the implementation.   </br>
++ Python 3.7 or above </br>
++ Pytorch 2.1.2  </br>
++ Tensorboard 2.12.1 </br>
++ Open-cv 4.10.0.84 </br>

# Dataset: 
Prepare banknotes in denominations from 1,000 VND to 500,000 VND. 
The data preparation will be through personal webcam by continuous image capture (Frame-by-Frame). Images will be captured and saved to folder with given label at speed of 30 frames per second (Webcam recording speed). There are 10 labels: 9 for Vietnamese currency denominations (1000 VND, 2000 VND, 5000 VND, 10000 VND, 20000 VND, 50000 VND, 100000 VND, 200000 VND, 500000 VND) and 1 label (0) for images with no money present. </br>

# Setting:
For more details, read the .docx file for our testing.




        
