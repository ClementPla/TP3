import cv2
import numpy as np
import math
"""
Data augmentation package. As it might be used for our personal project, all the operations are done on multichannel
images. But keep in mind that in the TP3, images are single-channel.
"""


def random_rotation(x, max_variation=5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                    border_value=0.0):
    """

    :param x: Input image, a tensor of shape (channel, row, col)
    :param max_variation: Maximum variational angle
    :param fixed_angle:
    :param interpolation: INTER_NEAREST,  INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
    :return: Transformed image
    """
    angle = np.random.uniform(low=-max_variation, high=max_variation)
    if x.ndim == 3:
        chan, rows, cols = x.shape
    elif x.ndim == 2:
        rows, cols = x.shape

    M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)

    dst = cv2.warpAffine(x, M, (cols, rows), flags=interpolation, borderMode=border_mode, borderValue=border_value)

    return dst


def random_translation(x, max_variation=5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                    border_value=0.0):

    tx = np.random.uniform(low=-max_variation, high=max_variation)
    ty = np.random.uniform(low=-max_variation, high=max_variation)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    if x.ndim == 3:
        chan, rows, cols = x.shape
    elif x.ndim == 2:
        rows, cols = x.shape

    dst = cv2.warpAffine(x, M, (cols, rows), flags=interpolation, borderMode=border_mode, borderValue=border_value)

    return dst


def random_shear(x, max_variation=15, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                 border_value=0.0):

    shear = math.radians((np.random.uniform(low=-max_variation, high=max_variation)))
    if x.ndim == 3:
        chan, rows, cols = x.shape
    elif x.ndim == 2:
        rows, cols = x.shape

    shear_matrix = np.array([[1.0, -np.sin(shear), 0.0],
                             [0.0, np.cos(shear), 0.0]])
    dst = cv2.warpAffine(x, shear_matrix, (cols, rows), flags=interpolation, borderMode=border_mode,
                         borderValue=border_value)

    return dst


def elastic_distortion(image, sigma, alpha, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                       border_value=0.0):

    random_dx = (np.random.rand(*image.shape)*2-1)
    random_dy = (np.random.rand(*image.shape)*2-1)
    smooth_dx = cv2.GaussianBlur(random_dx, ksize=None, sigmaX=sigma)*alpha
    smooth_dy = cv2.GaussianBlur(random_dy, ksize=None, sigmaX=sigma)*alpha
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='xy')
    map_x = np.reshape(x + smooth_dx, (28,28)).astype('float32')
    map_y = np.reshape(y + smooth_dy, (28,28)).astype('float32')

    dst = cv2.remap(image, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode,
                         borderValue=border_value)
    return dst



def vector2matrix(vector, shape):
    return np.reshape(vector, shape)


def matrix2vector(matrix):
    return matrix.flatten()

