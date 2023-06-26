import os
import torch
from dataset import Dataset
from functions import *
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np

################################################################################
# Daten laden
################################################################################
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

################################################################################
# Netz spezifizieren
################################################################################
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda:1'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

################################################################################
# Bestes trainiertes Modell laden
################################################################################
best_model = torch.load('./best_model.pth')

################################################################################
# Test-Datensatz
################################################################################
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

################################################################################
# ValidEpoch-Objekt erstellen um Metrikaussage über den Testdatensatz zu treffen
################################################################################
test_epoch = smp.utils.train.ValidEpoch(model=best_model, loss=loss, metrics=metrics, device=DEVICE)

logs = test_epoch.run(test_dataloader)

################################################################################
# Predictions visualisieren
################################################################################
test_dataset_vis = Dataset(x_test_dir, y_test_dir, classes=CLASSES)

for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualize_img_mask(image_vis, gt_mask, pr_mask, filename='test_' + str(i) + '.png')