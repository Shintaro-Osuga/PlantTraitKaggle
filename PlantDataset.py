import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    model_name = 'tf_efficientnetv2_b2'  # Name of pretrained classifier
    image_size = 224  # Input image size
    epochs = 12 # Training epochs
    batch_size = 8  # Batch size
    lr = 1e-4
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    num_folds = 5 # Number of folds to split the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['X4_mean', 'X11_mean', 'X18_mean',
                   'X26_mean', 'X50_mean', 'X3112_mean',]
    aux_class_names = list(map(lambda x: x.replace("mean","sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)

class PlantDataset(Dataset):
    def __init__(self, paths, features, labels=None, aux_labels=None, transform=None, augment=False):
        self.paths = paths
        self.features = features
        self.labels = labels
        self.aux_labels = aux_labels
        self.transform = transform
        self.augment = augment
        #these values are taken from the pd describe of the dataset manually
        self.means = torch.tensor([0.522942,	16.029181,	3.409873,	49.688374,	1.631341,	2046.341614], dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        feature = self.features[idx]

        # Read and decode image
        image = self.decode_image(path)
        # Apply augmentations
        if self.augment:
            augmented = self.transform(image=image)
            image = augmented['image']            
        else:
            # Ensure channel dimension is the first one
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)


        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            c_label = (label > self.means).long()
            
            aux_label = torch.tensor(self.aux_labels[idx])
            return {'images': image, 'features': feature}, (label, aux_label, c_label)
        else:
            return {'images': image, 'features': feature}

    def decode_image(self, path):
        image = Image.open(path)
        image = image.resize((CFG.image_size,CFG.image_size))
        image_mono = image.convert('1')
        image = np.array(image)
        # image = np.transpose(image, (2,0,1))
        image_mono = np.array(image_mono)
        image_mono = np.expand_dims(image_mono, axis=2)
        image_concat = np.concatenate((image, image_mono), axis=2)
        # print(f"image: {np.shape(image)} | image_mono: {np.shape(image_mono)} | image_concat: {np.shape(image_concat)}")
        return image_concat
