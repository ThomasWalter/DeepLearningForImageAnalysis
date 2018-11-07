'''Image data generator generating couples of images,
which can correspond to (image, operator(image)) or
(image, segmentation(image)).
Thus can be used to learn operators or segmentations.
'''
import numpy as np
import keras.backend as K

from dlia_tools.random_image_generator import DeadLeavesWithSegm


class RandomImageGeneratorBase(object):
    '''Base classe for generating 2D random images

    Arguments:
       dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
       (the depth) is at index 1, in 'tf' mode it is at index 3.
       It defaults to the `image_dim_ordering` value found in your
       Keras config file at `~/.keras/keras.json`.
       If you never set it, then it will be "th".
       image_augm: instance of image augmentation class
    '''

    def __init__(self, image_aug=None, dim_ordering=K.image_dim_ordering()):
        self.dim_ordering = dim_ordering
        self.image_aug = image_aug

    def flow(self):
        raise NotImplemented("flow method of RandomImageGeneratorBase is not implemented")


class DeadLeavesWithSegmGenerator(RandomImageGeneratorBase):
    '''Generate dead leaves model

    Arguments:
        Params:
        x_size, y_size: image dimensions
        rog_list: list of random object generators class instances
        noise: instance of noise generator class
        background_val: background value of images
        shuffle: are the random objects shuffled or sequentially drawn on the image (default)?
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension is at index 1, in 'tf' mode it is at index 3.
        It defaults to the `image_dim_ordering` value found in your Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "th".
        image_augm: instance of image augmentation class
        norm: normalization constant
    '''

    def __init__(self, x_size, y_size, rog_list, noise=None, background_val=0, shuffle=False, dim_ordering=K.image_dim_ordering(), image_augm=None, norm=255):
        self.__dead_leaves_w_segm__ = DeadLeavesWithSegm(x_size, y_size, rog_list, noise, background_val, shuffle, norm)
        super(DeadLeavesWithSegmGenerator, self).__init__(image_augm, dim_ordering)

    def flow(self, batch_size):
        return self.__dead_leaves_w_segm__.iterator(batch_size)
