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
    c = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def poisson_noise(x, severity=1):
    c = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='poisson', amount=c)
    return np.clip(x, 0, 1) * 255

def gaussian_noise(x, severity=1):
    c = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, severity=1):
    c = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255

def jpeg_compression(x, severity=1):
    c = [30, 25, 21, 18, 15, 12, 10, 8, 6, 3][severity - 1]

    output = BytesIO()
    x.convert('L').save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    return x
# /////////////// End Corruptions ///////////////
