import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class BBBC_039_data(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, convert_to_rgb=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.convert_to_rgb = convert_to_rgb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = np.array(Image.open(img_path))

        if self.convert_to_rgb:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.float32)
        mask = mask[..., 2]
        mask[mask == 255.0] = 1.0
        # get only the contour
        # mask = mask[..., 2]

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
