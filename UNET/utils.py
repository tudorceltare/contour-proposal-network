import os.path

import matplotlib.pyplot as plt
import torch
import torchvision
import albumentations as A
import numpy as np
import math
from albumentations.pytorch import ToTensorV2
from dataset import BBBC_039_data
from torch.utils.data import DataLoader
from datetime import datetime


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint at {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    convert_to_rgb=False,
):
    train_ds = BBBC_039_data(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        convert_to_rgb=convert_to_rgb,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BBBC_039_data(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        convert_to_rgb=convert_to_rgb,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            intersection = torch.logical_and(preds, y).sum()
            union = torch.logical_or(preds, y).sum()
            iou_score += (intersection + 1e-8) / (union + 1e-8)

    accuracy = num_correct / num_pixels
    dice_score /= len(loader)
    iou_score /= len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy * 100:.2f}")
    print(f"Dice score: {dice_score:.4f}")
    print(f"IoU score: {iou_score:.4f}")

    model.train()
    return accuracy, dice_score, iou_score


def split_and_process_img(img, h, w, model, transform, device='cuda'):
    original_h, original_w = img.shape[:2]

    rows = math.ceil(original_h / h)
    cols = math.ceil(original_w / w)

    pad_height = rows * h - original_h
    pad_width = cols * w - original_w

    # image should be np.array of shape (h, w)
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant')
    sub_images = padded_img.reshape(rows, h, cols, w)
    sub_images = sub_images.swapaxes(1, 2).reshape(-1, h, w)  # Reshape and combine

    sub_predictions = [get_prediction(sub_img, model=model, transform=transform, device=device) for sub_img in sub_images]

    # if sub_predictions is cuda tensor, convert it to numpy array
    if isinstance(sub_predictions[0], torch.Tensor):
        sub_predictions = [tensor.cpu().numpy() for tensor in sub_predictions]

    predictions_grid = np.array(sub_predictions).reshape(rows, cols, h, w)

    # Combine the processed sub-images to reconstruct the final image
    final_prediction = predictions_grid.swapaxes(1, 2).reshape(original_h + pad_height, original_w + pad_width)

    # Crop the final image to the original dimensions
    final_prediction = final_prediction[:original_h, :original_w]
    return final_prediction


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def get_prediction(image, model, transform, device='cuda'):
    model = model.to(device=device)
    model.eval()
    input_image = image
    if len(image.shape) == 2:
        input_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    input_image = torch.from_numpy(transform(image=input_image)['image']).unsqueeze(0).to(device=device)
    input_image = input_image.permute(0, 3, 1, 2)
    with torch.no_grad():
        prediction = model(input_image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()

    model.train()
    return prediction


def get_transforms(model_architecture='UNET', img_height=520, img_width=696):
    match model_architecture:
        case 'UNET':
            train_transform = A.Compose(
                [
                    A.Resize(height=img_height, width=img_width),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    # initial data is in range [0, 255], so we normalize to [0, 1] but keep the same distribution
                    A.Normalize(
                        mean=[0.0],
                        std=[1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2()
                ]
            )

            val_transform = A.Compose(
                [
                    A.Resize(height=img_height, width=img_width),
                    # initial data is in range [0, 255], so we normalize to [0, 1] but keep the same distribution
                    A.Normalize(
                        mean=[0.0],
                        std=[1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2()
                ]
            )
        case 'ResUNET':
            train_transform = A.Compose(
                [
                    A.RandomCrop(img_height, img_width),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.CoarseDropout(min_holes=17, max_holes=35, p=0.5),
                    A.GridDistortion(num_steps=4, distort_limit=0.3, p=0.3, border_mode=4),
                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.6), contrast_limit=0.1, p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2()
                ]
            )
            val_transform = A.Compose(
                [
                    A.RandomCrop(img_height, img_width),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2()
                ]
            )
        case _:
            raise ValueError(f'Invalid model architecture: {model_architecture}')
    return train_transform, val_transform


def plot_metrics(metrics_tuples_list, num_epochs, model, dir_path, save_plots=False):
    model.eval()
    epochs = np.arange(1, num_epochs + 1, 1)
    legend = []
    for metric_tuple in metrics_tuples_list:
        metric_values = metric_tuple[0]
        metric_values = np.array([tensor.item() for tensor in metric_values])
        plt.plot(epochs, metric_values)
        legend.append(metric_tuple[1])
    plt.grid()
    plt.xlabel('Epochs')
    plt.legend(legend, loc="lower right")
    if save_plots:
        plt.savefig(os.path.join(dir_path, 'metrics.png'))
    plt.show()

    model.train()


def print_device_details(device='cuda'):
    if device == 'cuda':
        if torch.cuda.device_count() >= 1:
            idx = torch.cuda.current_device()
            print(f'Currently running on: {torch.cuda.get_device_name(idx)}')