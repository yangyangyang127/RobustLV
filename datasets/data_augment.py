import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms

def randomShift(image):
    img = np.array(image)
    img0 = np.array(image)
    pos1 = random.uniform(0, 16)
    if pos1 < 8:
        pos = random.uniform(0, 16)
        if pos < 8:    # up
            xs = random.randint(1, 10)
            img[0:xs, :], img[xs:image.size[0], :] = img0[image.size[0]-xs:image.size[0], :], img0[0:image.size[0]-xs, :]
        else:    # down
            xs = random.randint(image.size[0]-10, image.size[0])
            img[0:xs, :], img[xs:image.size[0], :] = img0[image.size[0]-xs:image.size[0], :], img0[0:image.size[0]-xs, :]
    
    pos1 = random.uniform(0, 16)
    if pos1 < 8:
        pos = random.uniform(0, 16)
        if pos < 8:   # right
            ys = random.randint(1, 15)
            img[:, 0:ys], img[:, ys:image.size[0]] = img0[:, image.size[0]-ys:image.size[0]], img0[:, 0:image.size[0]-ys]
        else:    # left
            ys = random.randint(image.size[0]-10, image.size[0])
            img[:, 0:ys], img[:, ys:image.size[0]] = img0[:, image.size[0]-ys:image.size[0]], img0[:, 0:image.size[0]-ys]
    img1 = Image.fromarray(np.array(img, dtype='uint8'))
    return img1

def randomCrop(image):
    pos = random.uniform(0, 10)
    if pos < 5:
        ss = image.size[0]-4
        RandomCrop = transforms.RandomCrop(size=(ss, ss), padding=0)
        random_image = RandomCrop(image)
        random_image = transforms.Pad(padding=2)(random_image)
    elif pos > 5 and pos < 8:
        ss = image.size[0]-8
        RandomCrop = transforms.RandomCrop(size=(ss, ss), padding=0)
        random_image = RandomCrop(image)
        random_image = transforms.Pad(padding=4)(random_image)
    else:
        ss = image.size[0]-2
        RandomCrop = transforms.RandomCrop(size=(ss, ss), padding=0)
        random_image = RandomCrop(image)
        random_image = transforms.Pad(padding=1)(random_image)
    return random_image

def randomRotation(image):
    RR = transforms.RandomRotation(degrees=(-30, 30))
    rr_image = RR(image)
    return rr_image

def randomColorJitter(image):
    image1 = image.convert("RGB")
    RC = transforms.ColorJitter(brightness=0.8, contrast=0.8)
    rc_image1 = RC(image1)
    rc_image = rc_image1.convert('L')
    return rc_image

def DataAugmentation_per_chnl(image):
    im1 = image[:, :]
    x1min, x1max = im1.min(), im1.max()
    x1np = (im1 - x1min) / (x1max - x1min) * 255.0
    img1 = Image.fromarray(np.array(x1np, dtype='uint8'))
    possib = random.uniform(0, 10)
    if possib < 5:
        img1 = randomColorJitter(img1)
    possib = random.uniform(0, 10)
    if possib < 5:
        img1 = randomCrop(img1)
    possib = random.uniform(0, 10)
    if possib < 5:
        img1 = randomRotation(img1)
    possib = random.uniform(0, 10)
    if possib < 5:
        img1 = randomShift(img1)
    im1 = np.array([np.array(img1)], dtype="float32") / 255.0
    return im1

def DataAugmentation(images):
    img = np.zeros((images.shape[0], images.shape[1], images.shape[2]))
    for i in range(images.shape[0]):
        img[i, :, :] = DataAugmentation_per_chnl(images[i, :, :])     
    return img