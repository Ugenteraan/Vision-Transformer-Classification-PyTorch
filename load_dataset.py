
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import glob
from PIL import Image
import torch
from torch.utils.data import Dataset




class LoadLabelledDataset(Dataset):
    '''Loads the dataset from the given path. 
    '''

    def __init__(self, dataset_folder_path, image_size=224, image_depth=3, train=True, transforms=None, logger=None):
        '''Parameter Init.
        '''

        assert not dataset_folder_path is None, "Path to the dataset folder must be provided!"

        self.dataset_folder_path = dataset_folder_path
        self.transforms = transforms
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()
        self.logger = logger


    def get_classnames(self):
        '''Returns the name of the classes in the dataset.
        '''
        return os.listdir(f"{self.dataset_folder_path.rstrip('/')}/train/" )


    def read_folder(self):
        '''Reads the folder for the images with their corresponding label (foldername).
        '''

        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/train/"
        else:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/test/"

        for x in glob.glob(folder_path + "**", recursive=True):

            if not x.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            class_idx = self.classes.index(x.split('/')[-2])
            image_path_label.append((x, int(class_idx)))

        return image_path_label


    def __len__(self):
        '''Returns the total size of the data.
        '''
        return len(self.image_path_label)

    def __getitem__(self, idx):
        '''Returns a single image and its corresponding label.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.image_path_label[idx]

        try:

            image = Image.open(image_path).convert('RGB')

        except Exception as err:
            if self.logger is not None:
                self.logger.error(f"{image_path}")
                self.logger.error(f"Error loading image: {err}")
            


        if self.transforms:
            image = self.transforms(image)

        return {
            'images': image,
            'labels': label
        }