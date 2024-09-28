from PIL import Image
import numpy as np
from scipy import ndimage


def get_x(pos: float) -> str | None:
    if 45.0 <= pos <= 52.0:
        return "A"
    elif 178.0 <= pos <= 186.0:
        return "B"
    elif 310.0 <= pos <= 320.0:
        return "C"
    elif 445.0 <= pos <= 453.0:
        return "D"
    else:
        return None


def get_y(pos: float) -> int | None:
    if 45.0 <= pos <= 52.0:
        return 0
    elif 178.0 <= pos <= 186.0:
        return 1
    elif 310.0 <= pos <= 320.0:
        return 2
    elif 445.0 <= pos <= 453.0:
        return 3
    else:
        return None


def blob_doh(image, max_sigma=20, threshold=0.01):
    # Convert image to float
    image = image.astype(float)

    # Compute Gaussian kernels
    sigmas = np.linspace(1, max_sigma, 10)

    # Initialize the scale space
    scale_space = np.zeros(image.shape + (len(sigmas),))

    # Compute the DoH for each scale
    for i, sigma in enumerate(sigmas):
        # Compute Gaussian second derivatives
        gxx = ndimage.gaussian_filter(image, sigma, order=(0, 2))
        gyy = ndimage.gaussian_filter(image, sigma, order=(2, 0))
        gxy = ndimage.gaussian_filter(image, sigma, order=(1, 1))

        # Compute DoH response
        doh = sigma**2 * (gxx * gyy - gxy**2)

        # Store the response in scale space
        scale_space[:, :, i] = doh

    # Find local maxima in scale space
    local_max = ndimage.maximum_filter(scale_space, size=(3, 3, 3))
    mask = (scale_space == local_max) & (scale_space > threshold)

    # Get coordinates of peaks
    coordinates = np.column_stack(np.nonzero(mask))

    # Convert to (y, x, sigma) format
    blobs = []
    for coord in coordinates:
        y, x, scale_idx = coord
        sigma = sigmas[scale_idx]
        blobs.append((y, x, sigma))

    return np.array(blobs)


file = "datasets/input/1.png"
answer = ["D", "A", "D", "C"]

image = np.array(Image.open(file), dtype=np.float32)
blobs = blob_doh(image, max_sigma=20, threshold=0.01)

answers = ["", "", "", ""]

for idx, blob in enumerate(blobs):
    y, x, r = blob

    if r > 14 and r < 20 and len(answers) <= 4:
        img = Image.open(file)

        if (idx := get_y(y)) is not None:
            opt = get_x(x)
            if opt is not None:
                answers[idx] = opt

        bbox = (x - r, y - r, x + r, y + r)
        output = img.crop(bbox)

count = 0
for idx, x in enumerate(answers):
    if x == answer[idx]:
        count += 1

print(count)
