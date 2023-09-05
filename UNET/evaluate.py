import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils import load_checkpoint, split_and_process_img, split_and_process_loader_batch
from dataset import BBBC_039_data
from model import UNET, ResUNET
from tqdm import tqdm
from matplotlib import pyplot as plt


def evaluate_iou_threshold(loader, model, device='cuda'):
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


# evaluate the model on the test set using accuracy, IoU score, and Dice score.
# calculate the best score and the average score for the test set.
def evaluate_test_set(loader, model, device='cuda', model_architecture='UNET', transform=None):
    model.eval()
    accuracy_avg = 0.0
    iou_avg = 0.0
    dice_avg = 0.0
    for x, y in tqdm(loader):
        with torch.no_grad():
            if model_architecture == 'UNET':
                preds = torch.sigmoid(model(x.to(device)))
                preds = (preds > 0.5).float()
                # plt.title('Original')
                # plt.imshow(x.squeeze(0).squeeze(0).cpu().numpy())
                # plt.show()
                # plt.title('Prediction')
                # plt.imshow(preds.squeeze(0).squeeze(0).cpu().numpy())
                # plt.show()
                # return
            elif model_architecture == 'ResUNET':
                # print(f'x.shape: {x.shape}')
                # x.shape: torch.Size([batch_size, 3, h, w])
                # print(f'x.shape: {x.shape}')
                preds = split_and_process_loader_batch(x, 256, 256, model=model, device=device)
                preds = torch.from_numpy(preds).to(device).float()
            # calculate IoU score
            intersection = torch.logical_and(y.to(device), preds)
            union = torch.logical_or(y.to(device), preds)
            iou_score = (intersection.sum() + 1e-8) / (union.sum() + 1e-8)
            iou_avg += iou_score
            # calculate Dice score
            dice_score = (2 * (intersection.sum() + 1e-8)) / ((y.to(device) + preds).sum() + 1e-8)
            dice_avg += dice_score
            # calculate accuracy
            accuracy = (preds == y.to(device)).sum() / torch.numel(preds)
            accuracy_avg += accuracy
    accuracy_avg /= len(loader)
    iou_avg /= len(loader)
    dice_avg /= len(loader)
    print(f'Accuracy: {accuracy_avg:.4f}')
    print(f'IoU score: {iou_avg:.4f}')
    print(f'Dice score: {dice_avg:.4f}')
    model.train()
    return accuracy_avg, iou_avg, dice_avg


def main(model_architecture='UNET'):
    # Hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True
    TEST_IMG_DIR = '../dataset/BBBC_039_formatted/train/images'
    TEST_MASK_DIR = '../dataset/BBBC_039_formatted/train/boundary_labels'

    match model_architecture:
        case 'UNET':
            model = UNET(in_channels=1, out_channels=1).to(DEVICE)
            checkpoint = torch.load('checkpoints/binary_contour_150epochs_UNET.pth.tar', map_location=DEVICE)
            load_checkpoint(checkpoint, model)
            test_transform = A.Compose([
                # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0],
                    std=[1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ])
            test_ds = BBBC_039_data(
                image_dir=TEST_IMG_DIR,
                mask_dir=TEST_MASK_DIR,
                transform=test_transform,
            )
        case 'ResUNET':
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            model = ResUNET(encoder=encoder, out_channels=1).to(DEVICE)
            checkpoint = torch.load('checkpoints/binary_contour_1000epochs_ResUNET_AUGMENTED.pth.tar',
                                    map_location=DEVICE)
            # checkpoint = torch.load('checkpoints/binary_contour_350epochs_ResUNET_AUGMENTED.pth.tar',
            #                         map_location=DEVICE)
            # checkpoint = torch.load('checkpoints/binary_contour_350epochs_ResUNET.tar',
            #                         map_location=DEVICE)
            load_checkpoint(checkpoint, model)
            test_transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            test_ds = BBBC_039_data(
                image_dir=TEST_IMG_DIR,
                mask_dir=TEST_MASK_DIR,
                transform=test_transform,
                convert_to_rgb=True,
            )
        case _:
            raise NotImplementedError(f'Unknown model architecture: {model_architecture}')

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # evaluate_iou_threshold(test_loader, model, DEVICE)
    evaluate_test_set(test_loader, model, DEVICE, model_architecture, test_transform)


if __name__ == '__main__':
    main(model_architecture='ResUNET')
