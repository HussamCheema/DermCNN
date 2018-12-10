import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
print("yes")
print(os.getcwd())

data = []
labels = []
batch_size = 32
epochs = 10

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
diseaseFilesDirectory='/'

# for i in range(7):
i = 7
path=os.getcwd()+diseaseFilesDirectory+str(i)+'/'
print(path)
allfiles=os.listdir(path)
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]
for im in imlist:
    imarr=np.array(Image.open(path+im),dtype=np.float)
    imarr = imarr.reshape((1,)+imarr.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    i = 0
    for batch in datagen.flow(imarr, batch_size=1,save_to_dir= path+'GeneratedData', save_prefix='', save_format='jpg'):
        i += 1
        if i > 38:
            break  # otherwise the generator would loop indefinitely