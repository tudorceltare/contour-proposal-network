import gradio as gr
import cv2
import torch
import skimage.io
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
from UNET.model import UNET, ResUNET
from UNET.utils import load_checkpoint, split_and_process_img, split_and_process_loader_batch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_to_png(raw_img):
    percentile = 99.9
    high = np.percentile(raw_img, percentile)
    low = np.percentile(raw_img, 100 - percentile)

    img = np.minimum(high, raw_img)
    img = np.maximum(low, img)
    img = (img - low) / (high - low)
    img = skimage.util.img_as_ubyte(img)
    return img


def format_input_image(img, model_architecture):
    if isinstance(img, gr.inputs.Image):
        file_path = img.name
        if file_path.endswith(".tiff") or file_path.endswith(".tif"):
            img = np.array(img)
            img = convert_to_png(img)
        elif file_path.endswith(".png"):
            img = np.array(img)
        else:
            print("Input image format is not recognized.")
    else:
        print("Input image is not of the expected type.")
        img = np.array(img)

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
    np.repeat(img[:, :, np.newaxis], 3, axis=2)
    print(f'img.shape: {img.shape}')
    return img


def make_prediction(img, model_architecture):
    device = DEVICE
    img = format_input_image(img, model_architecture)
    match model_architecture:
        case 'UNET':
            model = models[0]["model"]
            transform = models[0]['transform']
            copy_img = np.copy(img)
            img = transform(image=img)['image']
            print(f'img.shape: {img.shape}')
            test_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()
            print(f'test_img.shape: {test_img.shape}')
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(test_img))
                preds = (preds > 0.5).float()
                preds = preds.squeeze(0).squeeze(0).squeeze(0).cpu().numpy()
            model.train()

            # if transform is not None:
            #     transformed = transform(image=img)
            #     img = transformed['image']

            negative_preds = 1.0 - preds

            overlapped_preds = np.stack((copy_img,) * 3, axis=-1)
            overlapped_preds[..., 2] = copy_img * negative_preds
            overlapped_preds[..., 0] = copy_img * negative_preds

            return overlapped_preds, preds

        case 'ResUNET':
            model = models[1]["model"]
            transform = models[1]["transform"]
            batch = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            batch = transform(image=batch)['image']
            batch = np.expand_dims(batch, axis=0)
            preds = split_and_process_loader_batch(batch, 256, 256, model=model, device=device)
            preds = preds.squeeze(0).squeeze(0)

            # from grayscale to rgb
            copy_preds = np.repeat(preds[:, :, np.newaxis], 3, axis=2)

            img_copy = np.copy(img)
            img_copy = np.repeat(img_copy[:, :, np.newaxis], 3, axis=2)

            negative_preds = 1.0 - preds
            overlapped_preds = img_copy
            overlapped_preds[..., 1] = img_copy[..., 1] * negative_preds
            overlapped_preds[..., 2] = img_copy[..., 2] * negative_preds

            return overlapped_preds, copy_preds

        case _:
            raise NotImplementedError(f'Unknown model architecture: {model_architecture}')


def load_models(device="cuda"):
    models = [
        {
            "name": "UNET",
            "model": UNET(in_channels=1, out_channels=1).to(device),
            "transform": A.Compose([
                A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ]),
            "checkpoint": "checkpoints/binary_contour_150epochs_UNET.pth.tar"
        },
        {
            "name": "ResUNET",
            "model": ResUNET(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False),
                             out_channels=1).to(device),
            "transform": A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "checkpoint": "checkpoints/binary_contour_1000epochs_ResUNET_AUGMENTED.pth.tar"
        }
    ]
    for model in models:
        load_checkpoint(torch.load(model["checkpoint"]), model["model"])
    return models


models = load_models(DEVICE)

if __name__ == '__main__':
    demo = gr.Interface(fn=make_prediction, inputs=[
        gr.inputs.Image(label="Input Image"),
        gr.inputs.Radio(["UNET", "ResUNET"], label="Model Selection"),
    ], outputs=[
        gr.outputs.Image(type='pil', label="Overlay Image"),
        gr.outputs.Image(type='pil', label="Predicted Mask"),
    ])
    demo.launch()