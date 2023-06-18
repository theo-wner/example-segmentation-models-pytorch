import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

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

# Hilfsfunktion, die mehrere Bilder in einem Subplot darstellt
# visualize(image_1=img1, image2=img2) kann dann über key und value iterieren
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
#-------------------------------------------------------------------------------
# Dataloader
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Augmentations
#-------------------------------------------------------------------------------


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


