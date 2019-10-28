"""Image data generator generating couples of images,
which can correspond to (image, operator(image)) or
(image, segmentation(image)).
Thus can be used to learn operators or segmentations.
"""
import numpy as np
import keras.backend as K

from dlia_tools.random_image_generator import DeadLeavesWithSegm


class RandomImageGeneratorBase(object):
    """Base classe for generating 2D random images

    Arguments:
       image_augm: instance of image augmentation class
       dim_ordering: 'channels_first' or 'channels_last'.
       It defaults to the `image_data_format` value found in your
       Keras config file at `~/.keras/keras.json`.
    """

    def __init__(self, image_aug=None, dim_ordering=K.image_data_format()):
        self.dim_ordering = dim_ordering
        self.image_aug = image_aug

    def flow(self):
        raise NotImplementedError(
            "flow method of RandomImageGeneratorBase is not implemented"
        )


class DeadLeavesWithSegmGenerator(RandomImageGeneratorBase):
    """Generate dead leaves model

    Arguments:
        Params:
        x_size, y_size: image dimensions
        rog_list: list of random object generators class instances
        noise: instance of noise generator class
        background_val: background value of images
        shuffle: are the random objects shuffled or sequentially drawn on the image (default)?
        dim_ordering: 'channels_first' or 'channels_last'.
        It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`.
        image_augm: instance of image augmentation class
        norm: normalization constant
    """

    def __init__(
        self,
        x_size,
        y_size,
        rog_list,
        noise=None,
        background_val=0,
        shuffle=False,
        dim_ordering=K.image_data_format(),
        image_augm=None,
        norm=255,
    ):
        self.__dead_leaves_w_segm__ = DeadLeavesWithSegm(
            x_size, y_size, rog_list, noise, background_val, shuffle, norm
        )
        super(DeadLeavesWithSegmGenerator, self).__init__(image_augm, dim_ordering)

    def flow(self, batch_size):
        return self.__dead_leaves_w_segm__.iterator(batch_size)
