import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# Klasse Dataset repräsentiert Datensatz und soll wie folgt abgreifbar sein: 
# dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])
# image, mask = Dataset[1]

class Dataset(BaseDataset):
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        # Instanzvariable: Liste aller Bildnamen
        self.ids = os.listdir(images_dir)
        # Instanzvariable: Liste aller Bildpfade
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # Instanzvariable: Liste aller Maskenpfade
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        # Instanzvariable: Liste aller vorkommenden Klassen als Index
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # Instanzvariable: Augmentation
        self.augmentation = augmentation
        # Instanzvariable: Preprocessing
        self.preprocessing = preprocessing
    
    # Funktion, die den Zugriff über den []-Operator regelt
    def __getitem__(self, i):
        # i-tes Bild einlesen
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Als RGB Bild einlesen
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE) # Als Graustufenbild einlesen
        
        # Erstelle für jede Klasse ein Bild mit
        #   1 - wenn der Pixel der Klasse angehört
        #   0 - wenn der Pixel nicht der Klasse angehört
        # Und reihe diese Bilder in einem Stack hintereinander
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)