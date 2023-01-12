import cv2
import os
from PIL import Image
import numpy as np
import time
from torchvision import transforms


def save_binary_img(x, iid, suffix):
    x = x.cpu().data.numpy()
    # print(x.shape)
    x = np.squeeze(x)
    # print(x.shape)
    img_save_dir = './img'
    x *= 255
    # timesign= time.strftime('%m%d_%H%M%S')
    im = Image.fromarray(x)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)

    if (im.mode == 'F'):
        im = im.convert('RGB')
    im.save(os.path.join(img_save_dir, str(iid) + '_' + suffix + '.jpg'))


def save_img(x, iid, suffix):
    img = x.cpu().clone()
    # print(x.shape)
    img = img.squeeze(0)
    # print(img.shape)
    img = transforms.ToPILImage()(img)
    img_save_dir = './img'
    # x *= 255
    # timesign= time.strftime('%m%d_%H%M%S')
    # im = Image.fromarray(x)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)

    # if(im.mode == 'F'):
    #   im = im.convert('RGB')
    img.save(os.path.join(img_save_dir, str(iid) + '_' + suffix + '.jpg'))

