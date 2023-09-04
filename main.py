import gradio as gr
from PIL import Image
import cv2
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def segment_image(model, image, device):
    model.eval()
    image_input = image.copy()
    image_input = image_input.transpose(2, 0, 1)  # Transpose to change channel position to first dimension
    image_input = torch.unsqueeze(torch.from_numpy(image_input), dim=0)  # add batch dimension
    image_input = image_input.to(device)
    with torch.no_grad():
        outputs = model(image_input)
    o = cd.asnumpy(outputs)
    num = len(o['contours'])
    s = int(np.ceil(np.sqrt(num)))
    plt.figure(None, (s * 24, s * 13.5))
    for idx in range(num):
        plt.subplot(s, s, idx + 1)
        cd.vis.show_detection(
            image=unmap(image),
            contours=o['contours'][idx],
            contour_linestyle='-',
            scores=o['scores'][idx],
            boxes=o['boxes'][idx],
        )
    plt.savefig('./results/output.png')
    return outputs


def open_image_from_path(file_path):
    image = cv2.imread(file_path)
    return image


def init_model_cpn(model_path, device):
    model = CpnU22(
        in_channels=1,
        order=6,  # higher umber means more complex shapes => higher fourier kernel
        samples=64,  # number of coordinates per contour
        refinement_iterations=3,
        nms_thresh=0.5,
        score_thresh=0.9,
        contour_head_stride=2,
        classes=2,
        refinement_buckets=6,
    ).to(device)
    cd.conf2tweaks_({'BatchNorm2d': {'momentum': 0.05}}, model)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    return model


def unmap(image):
    image = image * 255.0
    image = np.clip(image, 0, 255).astype('uint8')
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)
    return image


def normalize_image(image):
    # Convert image to floating point values
    image = image.astype(np.float32)

    # Normalize the image using min-max normalization
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

    return image_normalized


def run_demo(image):
    model = init_model_cpn('demos/checkpoint/cpn_epoch_10-cbf21cbecf4697fa.tar', 'cpu')
    image = normalize_image(image[..., 0:1])
    segment_image(model, image, 'cpu')
    return open_image_from_path('./results/output.png')


if __name__ == '__main__':
    # model = init_model_cpn('demos/checkpoint/cpn_epoch_10-cbf21cbecf4697fa.tar', 'cpu')
    # demo = gr.Interface(fn=run_cpn, inputs="image", outputs="image")
    # demo.launch()
    # image = open_image_from_path('dataset/BBBC_039/images/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78.tif')
    # image = normalize_image(image[..., 0:1])
    # segment_image(model, image, 'cpu')
    demo = gr.Interface(fn=run_demo, inputs="image", outputs="image")
    demo.launch()
