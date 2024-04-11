# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

# /////////////// Corruption Helpers ///////////////
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image as PILImage

# /////////////// Corruptions ///////////////

def impulse_noise(x, severity=1):
    c = [.03, .06, .10, 0.15, 0.20][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def poisson_noise(x, severity=1):
    c = [.03, .06, .10, 0.15, 0.20][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='poisson', amount=c)
    return np.clip(x, 0, 1) * 255

def gaussian_noise(x, severity=1):
    c = [.03, .06, 0.10, 0.15, 0.20][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 5][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255

def rician_noise(x, severity=1):
    v = [13, 13, 13, 13, 13][severity - 1]
    s = [10, 17, 24, 30, 36][severity - 1]
    N = x.size[0]*x.size[0]  # how many samples

    noise = np.random.normal(scale=s, size=(N, 2)) + [[v, 0]]
    noise = np.linalg.norm(noise, axis=1)

    x = np.array(x)
    x += noise.reshape(x.shape)
    return np.clip(x, 0, 255)

def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 5][severity - 1]

    output = BytesIO()
    print(x.size)
    x.convert('L').save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x
# /////////////// End Corruptions ///////////////
