import pandas as pd
import numpy as np
from scipy.misc import imread
import os

# convert image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Croping image to nxm size
def crop_center(img, n, m):
    # Get the shape of the image
    y, x = img.shape
    # Find the positions where you would like to crop the image
    start_x = x // 2-(n // 2)
    start_y = y // 2-(m // 2)
    # return the slices of the image you want
    return img[start_y:start_y+m, start_x:start_x+n]


# Process images in path file.
def createdataframe(path):
    frame = []
    for img in os.listdir(path):
        # lhs contains classifier 'person finger belongs to'
        lhs, rhs = img.split("_", 1)
        image = path + '/' + img
        im_data = imread(image)
        gray = rgb2gray(im_data)
        # testing different image crop sizes to see if it helps improve accuracy
        # gray = crop_center(gray, 220, 220)
        # ravel the image into ndarray and append the classifier append to list.
        x = gray.ravel()
        x = np.append(x, [int(lhs)])
        x.reshape(-1, len(x))
        frame.append(x)
    df = pd.DataFrame(frame)
    return df
