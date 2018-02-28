import tifffile as tf
import numpy as np

def imread(file):
    imageFile = tf.TiffFile(file)
    image_shape = imageFile.pages[0].shape
    if len(image_shape) == 3:
        return imageFile.asarray()
    if len(image_shape) == 2:
        y,x = image_shape
        image = np.zeros((len(imageFile.pages),y,x))
        for page in range(0,len(imageFile.pages)):
            image[page] = imageFile.asarray(key=page)
        return image

def imsave(file,data):
    data[data>65535] = 65535
    tf.imsave(file,data.astype(np.uint16))

