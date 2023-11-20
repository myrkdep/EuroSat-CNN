# EuroSAT Image Classification with CNN
This repository covers training a convolutional neural network (CNN) model on the EuroSAT dataset for multiclass land use classification.

## Dataset
The EuroSAT dataset contains 27,000 labeled satellite images covering 10 classes of land use in Europe:
* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake  
The images are in 3 bands (RGB) with a size of 64x64 pixels. The dataset was imported from Kaggle.

## Preprocessing
The splitfolders library was used to split the dataset into training (70%), validation (10%) and test (20%) sets.  

The Keras ImageDataGenerator was used to apply data augmentation to the training set for random rotations, zooms, flips etc. This expanded the diversity of data available for training.

## Model
A convolutional neural network (CNN) model was built using Keras and TensorFlow. The model architecture consists of:
* Conv2D layers for feature extraction
* MaxPooling2D for downsampling
* Dropout for regularization
* Dense layers for classification
* Adam optimizer
* Categorical crossentropy loss  
The model was trained for 40 epochs and an early stopping callback was used to prevent overfitting.

## Results
The model achieved 89% test accuracy after training.


## Future Work
* Experiment with additional data augmentation
* Try transfer learning
* Optimize hyperparameters like learning rate, batch size etc.
* Deploy model to production for land cover classification
  
## References
[EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset/data)
