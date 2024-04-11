import numpy as np
from PIL import Image
from corruptions import *

corruption_tuple = (gaussian_noise, poisson_noise,
                    impulse_noise, gaussian_blur, jpeg_compression, rician_noise)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}

def corrupt(x, severity=1, corruption_name=None, corruption_number=-1):

    if corruption_name:
        x_corrupted = corruption_dict[corruption_name](Image.fromarray(x), severity)
    elif corruption_number != -1:
        x_corrupted = corruption_tuple[corruption_number](Image.fromarray(x), severity)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(x_corrupted)
