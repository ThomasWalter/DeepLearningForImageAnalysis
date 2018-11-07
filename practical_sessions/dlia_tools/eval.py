'''Segmentation evaluation functions.
'''
import random

import numpy as np


def jaccard(im1, im2):
    """Computes Jaccard index between binary images im1 and im2.

    Arguments:
    im1, im2: numpy arrays containing positive integers.
    """

    union_vol = np.sum(np.maximum(im1, im2))
    if union_vol <= 0:
        raise ValueError("Images are empty (or contain negative values)")
    return(np.sum(np.minimum(im1, im2)) / union_vol)


def jaccard_curve(im_grey, im_bin):
    """Jaccard index computation for different thresholds.

    Arguments:
    im_grey: 8 bits images
    im_bin: binary image (only containing zeros and ones)"""

    values = []
    for grey in range(256):
        im_thresh = im_grey > grey
        values.append(jaccard(im_bin, im_thresh))

    return values

# def random_display_images_and_pred(X_in, Y_in, model, rows, cols, normalize=False, sigma=1, mu=0, norm_constant=1, lut_norm=True):

#     indices = np.array(random.sample(range(X_in.shape[0]), rows * cols))
#     X = X_in[indices]
#     Y_pred = model.predict(X)
#     Y = Y_in[indices]

#     if normalize:
#         X = X * sigma + mu
#     else:
#         X = norm_constant * X

#     if lut_norm:
#         lut_norm_f = Normalize()
#     else:
#         lut_norm_f = Normalize(vmin=0, vmax=255, clip=True)

#     X = np.squeeze(X)
#     fig, axarray = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(18, 12))
#     for r in range(rows):
#         for c in range(cols):
#             index = r + rows * c
#             axarray[r, c].set_title("Im %d gt=%.2f, pred=%.2f" % (
#                 indices[index],
#                 Y[index],
#                 Y_pred[index]
#             ))
#             axarray[r, c].imshow(X[index, :, :], cmap="gray", interpolation="nearest", norm=lut_norm_f)

#     # plt.colorbar()
#     plt.show()


# def random_display_images_and_value(X_in, Y_in, rows, cols, lut_norm=True):

#     indices = np.array(random.sample(range(X_in.shape[0]), rows * cols))
#     X = X_in[indices]
#     Y = Y_in[indices]

#     if lut_norm:
#         lut_norm_f = Normalize()
#     else:
#         lut_norm_f = Normalize(vmin=0, vmax=255, clip=True)

#     X = np.squeeze(X)
#     fig, axarray = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(18, 12))
#     for r in range(rows):
#         for c in range(cols):
#             index = r + rows * c
#             axarray[r, c].set_title("Im %d, val=%.2f" % (
#                 indices[index],
#                 Y[index]
#             ))
#             axarray[r, c].imshow(X[index, :, :], cmap="gray", interpolation="nearest", norm=lut_norm_f)

#     # plt.colorbar()
#     plt.show()
