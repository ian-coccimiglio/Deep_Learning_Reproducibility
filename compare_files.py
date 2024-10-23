#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 23:07:32 2024

@author: iancoccimiglio
"""

import numpy as np
from cellpose import models, io, metrics
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from logging import logging

im_path = "images/058_img.png"
label_1080_path = "images/gpu1080_labels.png"
label_4070_path = "images/gpu4070_labels.png"
logging.basicConfig(level=logging.INFO)
im = io.imread(im_path)


s[0, 0].imshow(rgb2gray(im), cmap="gray")
axs[0, 0].set_title("Original Image")
axs[1, 0].imshow(label_1080_path, cmap="viridis")
axs[1, 0].set_title("GPU Mask")
axs[1, 1].imshow(label_4070_path, cmap="viridis")
axs[1, 1].set_title("CPU Mask")

m1 = gpu_mask > 0 # Converts label image to binary mask
m2 = cpu_mask > 0

axs[0, 1].imshow(m1 != m2, cmap="gray")
axs[0, 1].set_title("Difference in Masks")

plt.tight_layout()