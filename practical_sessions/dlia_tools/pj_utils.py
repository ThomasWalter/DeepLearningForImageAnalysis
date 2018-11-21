import numpy as np
import os
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
def load_tissue_segmentation_data(path, prefix="train", size=(224, 224)):
    files = glob(os.path.join(path, prefix, "jpg", "*.jpg"))
    n_samples = len(files)
    X = np.zeros(shape=(n_samples, size[0], size[1], 3), dtype="float32")
    Y = np.zeros(shape=(n_samples, size[0], size[1], 1), dtype="uint8")
    for i, f in enumerate(files):
        X[i] = resize(imread(f), size, order=1, preserve_range=True)
        f = f.replace("jpg", "lbl")
        f = f.replace(".lbl", ".jpg")
        y = imread(f)
        y[y > 128] = 255
        y[y < 129] = 0
        Y[i,:,:,0] = resize(y, size, order=0, preserve_range=True).astype(int)
    Y[Y > 0] = 1    
    return X, Y

def rot90(m):
    return np.rot90(m, k=1)

def rot180(m):
    return np.rot90(m, k=2)

def rot270(m):
    return np.rot90(m, k=3)

def id(m):
    return m

def augment_images(array, array_gt):
    operations = [id, np.fliplr, np.flipud, rot90, rot180, rot270]
    m_samples, mx, my, mc = array.shape
    res = np.zeros(shape=(m_samples * len(operations), mx, my, mc))
    res_gt = np.zeros(shape=(m_samples * len(operations), mx, my, mc))
    for i in range(m_samples):
        for k, operation in enumerate(operations):
            res[i + k * m_samples] = operation(array[i])
            res_gt[i + k * m_samples] = operation(array_gt[i])
    return res, res_gt

