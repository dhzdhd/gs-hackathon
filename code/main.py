import os
from PIL import Image
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt

# Input format
# omr_sheet_1.png
# D A D C

# file = input()
# answer = input().split()
file = "datasets/input/1726807522-7153d428fa-omr_sheet_1.png"
answer = ["D", "A", "D", "C"]

image = np.array(Image.open(file), dtype=np.float32)

blobs = blob_doh(image, max_sigma=30, threshold=0.01)


fig, axes = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
ax = axes


ax.imshow(image)
for idx, blob in enumerate(blobs):

    y, x, r = blob
    c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
    ax.add_patch(c)
ax.set_axis_off()
