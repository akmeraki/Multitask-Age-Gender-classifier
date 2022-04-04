# Multitask-Age-Gender-classifier
This project was created to built a streamlit application that predicts Age and gender provided an imput Image. 
To build the model we create a Multitask network to classify Age category and Gender. 


## Requirements
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [Streamlit](https://streamlit.io/) (An open-source app framework for Machine Learning and Data Science teams)

## Table of Contents
-  [Folder Structure](#Folder-Structure)
-  [Preprocessing](#Preprocessing) 
-  [Train a model](#Train-a-model)
-  [Model Architecture](#Model-Architecture)
-  [Future Work](#future-work)
-  [Acknowledgments](#acknowledgments)

## Folder Structure
```
├──  data    - folder where the data & meta data is stored
│    └── aligned  - Image dataset  
|    └── data_dict.json - json file the contains lables for the data 
│  
├── models            - this folder contains any model of your project.
│   └──  AWS           - this folder is for model trained on AWS  
│   └──  local         - this folder is for model trained on local machine 
│
├── src
│    └── dataset_.py   - Contains the Pytorch Dataset Class for the dataset
│    └── model_.py    - Contains all the models used (CNN_3_layer, ResNet18, MobileNet_V2)
│    └── utility_.py  - Contains all utility functions useful for exploration, training and evaluation
|    └── main.py      - File that contains the full training functions to train using command line 
|
├── AGE_gender_classifier_EDA.ipynb   - This Jupyter notebook is for Exploratory Data Analysis
| 
├── Training_Gender_Age_classifier.ipynb   - This Jupyter notebook is for Training the Multi-task Classifier locally
|
├── Testing_Gender_Age_classifier.ipynb   - This Jupter notebook is for Training the Multi-task Classifier models locally 
```

## Preprocessing

## Train a model


## Future Work 




## Acknowledgements 


