import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils import load_checkpoint
from dataset import BBBC_039_data
from model import UNET


def evaluate(loader, model, device='cuda'):
    model.eval()

    # calculate the best threshold for the model
    best_threshold = 0.3
    best_iou_score = 0.0
    for threshold in np.arange(0.3, 1.0, 0.05):
        iou_avg = 0.0
        for x, y in loader:
            with torch.no_grad():
                preds = torch.sigmoid(model(x.to(device)))
                preds = (preds > threshold).float()
                # calculate IoU score
                intersection = torch.logical_and(y.to(device), preds)
                union = torch.logical_or(y.to(device), preds)
                iou_avg += (intersection.sum() + 1e-8) / (union.sum() + 1e-8)
        iou_avg /= len(loader)
        if iou_avg > best_iou_score:
            best_iou_score = iou_avg
            best_threshold = threshold
        print(f'Threshold: {threshold:.2f}, IoU score: {iou_avg:.4f}')
    print(f'Best threshold: {best_threshold:.2f} with IoU score: {best_iou_score:.4f}')

    model.train()


def main():
    # Hyperparameters
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    BATCH_SIZE = 16
    # BATCH_SIZE = 32
    # BATCH_SIZE = 64
    NUM_EPOCHS = 150
    SAVE_CHECKPOINT_EVERY = 5
    NUM_WORKERS = 2
    # IMAGE_HEIGHT = 160  # 520 originally
    IMAGE_HEIGHT = 520  # 520 originally
    # IMAGE_WIDTH = 240   # 696 originally
    IMAGE_WIDTH = 696  # 696 originally
    PIN_MEMORY = True
    LOAD_MODEL = False
    TRAIN_IMG_DIR = '../dataset/BBBC_039_formatted/train/images'
    TRAIN_MASK_DIR = '../dataset/BBBC_039_formatted/train/boundary_labels'
    VAL_IMG_DIR = '../dataset/BBBC_039_formatted/val/images'
    VAL_MASK_DIR = '../dataset/BBBC_039_formatted/val/boundary_labels'
    TEST_IMG_DIR = '../dataset/BBBC_039_formatted/test/images'
    TEST_MASK_DIR = '../dataset/BBBC_039_formatted/test/boundary_labels'

    test_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )

    test_ds = BBBC_039_data(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MASK_DIR,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    checkpoint = torch.load('checkpoints/my_checkpoint.pth.tar')
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    load_checkpoint(checkpoint, model)

    evaluate(test_loader, model, DEVICE)


if __name__ == '__main__':
    main()
