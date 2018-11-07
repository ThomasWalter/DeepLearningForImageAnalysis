import numpy as np


def draw_disk(im_in, x, y, r, v):
    """Draw disk.

    Draw disk at given position with given radius and grey level
    value on the input image im_in. Other image values are not modified.

    Params:
    im_in (2D numpy array): input image, where the disk will be drawn.
    x, y: coordinates of ring center.
    r: disk radius
    v: grey level value of ring.
    """
    (x_size, y_size) = im_in.shape
    r_2 = r**2
    min_x = int(max(0, np.floor(x - r)))
    max_x = int(min(x_size, np.ceil(x + r)))
    min_y = int(max(0, np.floor(y - r)))
    max_y = int(min(y_size, np.ceil(y + r)))
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            tmp = (x - i) ** 2 + (y - j) ** 2
            if tmp < r_2:
                im_in[i, j] = v


def draw_ring(im_in, x, y, r1, r2, v):
    """Draw ring.

    Draw ring at given position with given external and internal radius
    and grey level value on the input image im_in. Other image values are
    not modified.

    Params:
    im_in (2D numpy array): input image, where the ring will be drawn.
    x, y: coordinates of ring center.
    r1, r2: external and internal radii, respectively.
    v: grey level value of ring.
    """
    (x_size, y_size) = im_in.shape
    r1_2 = r1**2
    r2_2 = r2**2
    min_x = int(max(0, np.floor(x - r1)))
    max_x = int(min(x_size, np.ceil(x + r1)))
    min_y = int(max(0, np.floor(y - r1)))
    max_y = int(min(y_size, np.ceil(y + r1)))
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            r_2 = (x - i) ** 2 + (y - j) ** 2
            if r_2 < r1_2 and r_2 >= r2_2:
                im_in[i, j] = v


class RandomIntGenUniform(object):
    """Random integer value generator with uniform distribution.

    Parameters:
    mini: minimal value (inclusive).
    maxi: maximal value (exclusive).
    """

    def __init__(self, mini, maxi):
        self.__mini__ = mini
        self.__maxi__ = maxi

    def __call__(self):
        return np.random.randint(self.__mini__, self.__maxi__)


class RandomPosGenUniform(object):
    """Random 2D positive generator with uniform distribution."""

    def __init__(self, x_max, y_max, x_min=0, y_min=0):
        (self.__y_max__, self.__x_max__) = (x_max, y_max)
        (self.__y_min__, self.__x_min__) = (x_min, y_min)

    def __call__(self, shape=None):
        if shape is None:
            return (np.random.randint(self.__x_min__, self.__x_max__), np.random.randint(self.__y_min__, self.__y_max__))
        else:
            return (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))


class RandomPosGenConstant(object):
    """Random 2D positive generator given constant values."""
    def __init__(self, x, y):
        (self.__ysize__, self.__xsize__) = (x, y)

    def __call__(self):
        return (self.__ysize__, self.__xsize__)


class ROG_disks(object):
    """Random object generator: disks.

    Params:
    rig_number: random integer generator class instance - number of disks
    rpg: random position generator class instance
    rig_radius: random integer generator class instance used for radius value
    rig_val: random integer generator class instance - grey level value of each disk
    gt: 0 means the object will not appear in the ground truth segmentation. Otherwise, 1 or more
        is the label value.
    """

    def __init__(self, rig_number, rpg, rig_radius, rig_val, gt=1):
        self.__rig_number__ = rig_number
        self.__random_pos_gen__ = rpg
        self.__rig_radius__ = rig_radius
        self.__rig_val__ = rig_val
        self.__gt__ = gt

    def __call__(self, im, segm):
        for i in range(self.__rig_number__()):
            self.single(im, segm)

    def single(self, im, segm):
        """Single disk generation."""
        (x, y) = self.__random_pos_gen__(im.shape)
        r = self.__rig_radius__()
        v = self.__rig_val__()
        draw_disk(im, x, y, r, v)
        if self.__gt__ > 0:
            draw_disk(segm, x, y, r, self.__gt__)


class ROG_rings(object):
    """Random object generator: rings.

    Params:
    rig_number: random integer generator class instance - number of rings
    rpg: random position generator class instance
    rig_radius: random integer generator class instance - radius
    rig_val: random integer generator class instance - grey level value of each ring
    gt: 0 means the object will not appear in the ground truth segmentation. Otherwise, 1 or more
        is the label value.
    rat_ratio: ration between internal and external radii.
    """

    def __init__(self, rig_number, rpg, rig_radius, rig_val, gt=1, rad_ratio=0.5):
        self.__rig_number__ = rig_number
        self.__random_pos_gen__ = rpg
        self.__rig_radius__ = rig_radius
        self.__rig_val__ = rig_val
        self.__gt__ = gt
        self.__ratio__ = rad_ratio

    def __call__(self, im, segm):
        for i in range(self.__rig_number__()):
            self.single(im, segm)

    def single(self, im, segm):
        """Single ring generation."""
        (x, y) = self.__random_pos_gen__(im.shape)
        r1 = self.__rig_radius__()
        r2 = r1 * self.__ratio__
        v = self.__rig_val__()
        draw_ring(im, x, y, r1, r2, v)
        if self.__gt__ > 0:
            draw_ring(segm, x, y, r1, r2, self.__gt__)


class RandomObject(object):
    def __init__(self):
        self.__name__ = None
        self.__descriptors_list__ = None

    def get_name(self):
        if self.__name__ is None:
            raise(Exception, "Error name is not set.")
        return self.__name__

    def get_descriptors_names(self):
        if self.__descriptors_list__ is None:
            raise(Exception, "Descriptors list is None.")
        return self.__descriptors_list__


class AdditiveGaussianNoise(RandomObject):
    """Add gaussian noive of standard deviation sigma to input image.

    Params:
    sigma: standard deviation of Gaussian noise.
    """

    def __init__(self, sigma):
        RandomObject.__init__(self)
        self.__sigma__ = sigma
        self.__name__ = "AdditiveGaussianNoise"
        self.__descriptors_list__ = ["sigma"]

    def __call__(self, im_in):
        im_prov = im_in.astype('float')
        noise = np.array(self.__sigma__ * np.random.randn(im_in.shape[0], im_in.shape[1]))
        im_prov += noise
        im_prov[im_prov < 0] = 0
        im_prov[im_prov > 255] = 255
        im_in[:, :] = im_prov[:, :]
        return [self.__sigma__, ]


class DeadLeavesWithSegm(object):
    """
    Params:
    x_size, y_size: image dimensions
    rog_list: list of random object generators class instances
    noise: instance of noise generator class
    background_val: background value of images
    shuffle: are the random objects shuffled, or are drawn sequentially on the image (default)?
    norm: normalization constant
    """
    def __init__(self, x_size, y_size, rog_list, noise=None, background_val=0, shuffle=False, norm=255):
        self.__x__ = x_size
        self.__y__ = y_size
        self.__list__ = rog_list
        self.__noise__ = noise
        self.__bg__ = background_val
        self.__shuffle__ = shuffle
        self.__norm__ = norm

    def draw(self, im, segm):
        if self.__shuffle__ is False:
            for rog in self.__list__:
                rog(im, segm)
        else:
            raise NotImplemented("True shuffle is not yet implemented by DeadLeavesWithSegm")
        if self.__noise__ is not None:
            self.__noise__(im)

    def __call__(self, number=1):
        out_dict = {}
        out_dict["images"] = []
        out_dict["segm"] = []
        for im_i in range(number):
            im = np.empty([self.__y__, self.__x__], dtype='uint8')
            segm = np.zeros([self.__y__, self.__x__], dtype='uint16')
            im.fill(self.__bg__)
            self.draw(im, segm)
            out_dict["images"].append(im)
            out_dict["segm"].append(segm)
        return out_dict

    def iterator(self, batch_size=1):
        batch_x = np.zeros((batch_size, self.__x__, self.__y__, 1), dtype='float32')
        batch_x[:, :, :, :] = self.__bg__
        batch_y = np.zeros((batch_size, self.__x__, self.__y__, 1), dtype='float32')

        while(True):
            for i in range(batch_size):
                self.draw(batch_x[i, :, :, 0], batch_y[i, :, :, 0])
            batch_x /= self.__norm__
            yield (batch_x, batch_y)
            batch_x[:, :, :, :] = self.__bg__
            batch_y[:, :, :, :] = 0
