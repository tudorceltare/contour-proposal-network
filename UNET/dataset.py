import os
from os.path import join
from imageio import imread
import cv2
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage import measure
from skimage.measure import regionprops


# def read_all(directory, filename):
#     if filename is None or directory is None:
#         return None
#     with open(join(directory, filename), 'r') as f:
#         return [i.strip() for i in f.readlines()]
#
#
# def load(images_directory, masks_directory, names, **label_kwargs):
#     if None in (images_directory, masks_directory):
#         return None, None, None
#     images = [imread(join(images_directory, f.replace('.png', '.tif'))) for f in names]
#     masks = [imread(join(masks_directory, f)) for f in names]
#     return images, masks, [measure.label(m[:, :, 0], **label_kwargs) for m in masks]
#
# def labels2contours(labels, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE, flag_fragmented_inplace=False,
#                     raise_fragmented=True, constant=-1) -> dict:
#     """Labels to contours.
#
#     Notes:
#         - If ``flag_fragmented_inplace is True``, ``labels`` may be modified inplace.
#
#     Args:
#         labels:
#         mode:
#         method: Contour method. CHAIN_APPROX_NONE must be used if contours are used for CPN.
#         flag_fragmented_inplace: Whether to flag fragmented labels. Flagging sets labels that consist of more than one
#             connected component to ``constant``.
#         constant: Flagging constant.
#         raise_fragmented: Whether to raise ValueError when encountering fragmented labels.
#
#     Returns:
#         dict
#     """
#     crops = []
#     contours = OrderedDict()
#     for channel in np.split(labels, labels.shape[2], 2):
#         crops += [(p.label, p.image) + p.bbox[:2] for p in regionprops(channel)]
#     for label, crop, oy, ox in crops:
#         crop.dtype = np.uint8
#         r = cv2.findContours(crop, mode=mode, method=method, offset=(ox, oy))
#         if len(r) == 3:  # be compatible with both existing versions of findContours
#             _, c, _ = r
#         elif len(r) == 2:
#             c, _ = r
#         else:
#             raise NotImplementedError('try different cv2 version')
#         try:
#             c, = c
#         except ValueError as ve:
#             if flag_fragmented_inplace:
#                 labels[labels == label] = constant
#             elif raise_fragmented:
#                 raise ValueError('Object labeled with multiple connected components.')
#             continue
#         if len(c) == 1:
#             c = np.concatenate((c, c), axis=0)  # min len for other functions to work properly
#         contours[label] = c
#     if labels.shape[2] > 1:
#         return OrderedDict(sorted(contours.items()))
#     return contours
#
#
# class _BBBC039:
#     def __init__(self, directory, mode: str):
#         assert mode in ('train', 'test', 'val')
#
#         meta_directory = join(directory, 'metadata')
#         masks_directory = join(directory, 'masks')
#         images_directory = join(directory, 'images')
#
#         self.names = read_all(meta_directory, {
#             'train': 'training.txt',
#             'val': 'validation.txt',
#             'test': 'test.txt'
#         }[mode])
#
#         self.images, self.masks, self.labels = load(images_directory, masks_directory, self.names)
#
#     def plot(self, num=1, figsize=(20, 15)):
#         for i in np.random.randint(0, len(self), num):
#             show_detection(image=self.images[i], contours=labels2contours(self.labels[i]),
#                            figsize=figsize, contour_linestyle='-')
#
#     def __getitem__(self, item):
#         return self.names[item], self.images[item], self.masks[item], self.labels[item]
#
#     def __len__(self):
#         return len(self.images)
#
#
# class BBBC039Train(_BBBC039):
#     def __init__(self, directory):
#         super().__init__(directory, mode='train')
#
#
# class BBBC039Val(_BBBC039):
#     def __init__(self, directory):
#         super().__init__(directory, mode='val')
#
#
# class BBBC039Test(_BBBC039):
#     def __init__(self, directory):
#         super().__init__(directory, mode='test')

class BBBC_039_data(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
