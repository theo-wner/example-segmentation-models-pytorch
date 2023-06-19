import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from dataset import Dataset
from functions import *
import torch

################################################################################
# Daten laden
################################################################################
#-------------------------------------------------------------------------------
# Daten ins Programm laden
#-------------------------------------------------------------------------------
DATA_DIR = './data/CamVid/'

if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')

# Ordnerpfade der jeweiligen Subsets abgreifen
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

#-------------------------------------------------------------------------------
# Dataloader
#-------------------------------------------------------------------------------
dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])

# Testweise visualisieren
image, mask = dataset[4]
visualize(image=image, cars_mask=mask.squeeze())

#-------------------------------------------------------------------------------
# Augmentations
#-------------------------------------------------------------------------------
# Vor dem Abgreifen eines Bildes über den []-Operator soll dieses durch eine 
# Pipeline an zufälligen Trafos geschickt werden

augmented_dataset = Dataset(x_train_dir, y_train_dir, 
                            augmentation=get_training_augmentation(), 
                            classes=['car'])

# Testweise dasselbe Bild 3 Mal visualisieren 
# --> 3 Unterschiedliche Bilder dank zufälligen Trafos
for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(filename=str(i), image=image, mask=mask.squeeze())

################################################################################
# Modell erstellen
################################################################################

################################################################################
# Trainieren
################################################################################

################################################################################
# Testen
################################################################################

################################################################################
# Predictions visualisieren
################################################################################


