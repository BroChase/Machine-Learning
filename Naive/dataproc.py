import pandas as pd
import numpy as np
from scipy.misc import imread
import os

def createdataframe(path, path2):
    frame = []
    # get image(19x19) from folder one at a time and process them into 361 ndarray
    # append a 1'true' to the array because it is a face
    # reshape the array and push to frame []
    for img in os.listdir(path):
        image = path + '/' + img
        imData = imread(image)
        x = imData.ravel()
        x = np.append(x, [1])
        x.reshape(-1, len(x))
        frame.append(x)
    # Create pandas dataframe from all the face data in frame[]
    df = pd.DataFrame(frame)

    frame = []
    for img in os.listdir(path2):
        image = path2 + '/' + img
        imData = imread(image)
        x = imData.ravel()
        x = np.append(x, [0])
        x.reshape(-1, len(x))
        frame.append(x)
    # Create pandas dataframe from all the non-face data in frame[]
    df2 = pd.DataFrame(frame)
    # Combine dataframes into one
    result = df.append(df2)
    return result

