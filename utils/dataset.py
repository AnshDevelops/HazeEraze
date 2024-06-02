import os
from torch.utils.data import Dataset
from PIL import Image


class ResideDataset(Dataset):
    def __init__(self, root_dir, dataset_type='ITS', category='indoor', mode='train', transform=None):
        """
        Initialize the dataset loader.
        Args:
            root_dir (string): Base directory containing the dataset.
            dataset_type (string): 'ITS' for training or 'SOTS' for testing.
            category (string): 'indoor' or 'outdoor', used only if dataset_type is 'SOTS'.
            mode (string): 'train' or 'test', to adjust paths based on the dataset type.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.transform = transform
        if mode == 'train' and dataset_type == 'ITS':
            self.hazy_dir = os.path.join(root_dir, 'hazy')
            self.clear_dir = os.path.join(root_dir, 'clear')
        elif mode == 'test' and dataset_type == 'SOTS':
            self.hazy_dir = os.path.join(root_dir, category, 'hazy')
            self.clear_dir = os.path.join(root_dir, category, 'clear')

        self.hazy_images = [os.path.join(self.hazy_dir, f) for f in os.listdir(self.hazy_dir) if f.endswith('.jpg')]
        self.clear_images = [os.path.join(self.clear_dir, f) for f in os.listdir(self.clear_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = self.hazy_images[idx]
        clear_image_path = self.clear_images[idx]

        hazy_image = Image.open(hazy_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        return {'hazy': hazy_image, 'clear': clear_image}
