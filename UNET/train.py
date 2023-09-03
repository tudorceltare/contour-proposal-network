import torch
import albumentations as A
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET, ResUNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    plot_metrics,
    print_device_details,
    split_and_process_img,
    get_transforms,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
BATCH_SIZE = 16
# BATCH_SIZE = 32
# BATCH_SIZE = 64
NUM_EPOCHS = 1000
SAVE_CHECKPOINT_EVERY = 5
NUM_WORKERS = 2
# IMAGE_HEIGHT = 160  # 520 originally
# IMAGE_HEIGHT = 520  # 520 originally
IMAGE_HEIGHT = 256  # 520 originally
# IMAGE_WIDTH = 240   # 696 originally
# IMAGE_WIDTH = 696   # 696 originally
IMAGE_WIDTH = 256   # 696 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '../dataset/BBBC_039_formatted/train/images'
TRAIN_MASK_DIR = '../dataset/BBBC_039_formatted/train/boundary_labels'
VAL_IMG_DIR = '../dataset/BBBC_039_formatted/val/images'
VAL_MASK_DIR = '../dataset/BBBC_039_formatted/val/boundary_labels'
TRANSFORM_TO_PNG = False


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        loop.set_postfix(loss=loss.item())


def main(model_architecture='UNET'):
    print_device_details(DEVICE)
    # transform for image with multiple masks
    train_transform, val_transform = get_transforms(model_architecture, IMAGE_HEIGHT, IMAGE_WIDTH)

    if model_architecture == 'UNET':
        model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    elif model_architecture == 'ResUNET':
        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = ResUNET(encoder, out_channels=1, freeze_encoder=True).to(DEVICE)
    else:
        raise ValueError(f'Unknown model architecture: {model_architecture}')

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        convert_to_rgb=(model_architecture == 'ResUNET'),
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    accuracy_list = []  # List to store accuracy values
    dice_score_list = []  # List to store Dice scores
    iou_score_list = []  # List to store IoU scores
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # save model every SAVE_CHECKPOINT_EVERY epochs
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint, filename="checkpoints/my_checkpoint.pth.tar")
            # print some examples to a folder
            # TODO: change this function to save only one image
            save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

        # check accuracy
        accuracy, dice_score, iou_score = check_accuracy(val_loader, model, device=DEVICE)

        accuracy_list.append(accuracy)
        dice_score_list.append(dice_score)
        iou_score_list.append(iou_score)

    # final save
    if NUM_EPOCHS % SAVE_CHECKPOINT_EVERY != 0:
        save_checkpoint(checkpoint, filename="checkpoints/my_checkpoint.pth.tar")
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

    metric_tuples = [
        (accuracy_list, 'Accuracy'),
        (dice_score_list, 'Dice Score'),
        (iou_score_list, 'IoU')
    ]

    plot_metrics(metric_tuples, NUM_EPOCHS, model, dir_path='plots/', save_plots=True)


if __name__ == "__main__":
    main(model_architecture='ResUNET')
