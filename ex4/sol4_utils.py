from scipy.signal import convolve2d
import numpy as np
from imageio import imwrite, imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve as _convolve

# ----------- Constants -----------
MIN_IMG_SIZE_IN_EACH_AXIS = 16
EXPEND_FACTOR = 2


def _im_downsample(im, blur_filter):
    """
    downsamples the image according to the blur filter given and takes
    every even pixel in the image.
    :param im: np.array image
    :param blur_filter: np.array of size (1,) of the filter
    :return: smaller sample.
    """
    im = _convolve(im, blur_filter, mode='reflect')
    im = _convolve(im, blur_filter.T, mode='reflect')
    im = im[::2, ::2]
    return im


def _im_expand(im, blur_filter):
    """
    expands the image accodring to the blur filter. Makes the image twice
    as bigger.
    :param im: np.array image
    :param blur_filter: np.array of size (1,) of the filter
    :return: bigger image.
    """
    expended_im = np.zeros(
        (im.shape[0] * EXPEND_FACTOR, im.shape[1] * EXPEND_FACTOR))
    expended_im[1::2, 1::2] = im
    blur_filter = blur_filter * EXPEND_FACTOR
    expended_im = _convolve(_convolve(expended_im, blur_filter),
                            blur_filter.T)
    return expended_im

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

def read_image(filename, representation):
    """
    function which reads an image le and converts it into a given
    representation.
    :param filename: the filename of an image on disk (could be grayscale or
    RGB).
    :param representation: representation code, either 1 or 2 defining
    whether the output should be a grayscaleimage (1) or an RGB image (2).
    If the input image is grayscale, we won't call it with representation = 2.
    :return: rgb or grayscale img
    """
    image = imread(filename)
    new_image = image.astype(np.float64)
    new_image /= 255
    if representation == 1:
        new_image = rgb2gray(new_image)
    return new_image

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    constructs a gaussian pyramid.
    :param im: grayscale image with double values in [0,1].
    :param max_levels: maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the gaussian filter
    :return: pyr, filter_vec. pyr - python array where the elements are
    images with different sizes. filter_vec - the gaussian filter used.
    """
    blur_filter_not_normalized = _get_binomial_coefficients(filter_size)
    blur_filter = blur_filter_not_normalized / np.sum(
      blur_filter_not_normalized)

    pyr = list()
    curr_im = im
    while ((curr_im.shape[0] >= MIN_IMG_SIZE_IN_EACH_AXIS) and
           (curr_im.shape[1] >= MIN_IMG_SIZE_IN_EACH_AXIS) and
           (len(pyr) < max_levels)):
      pyr.append(curr_im)
      curr_im = _im_downsample(curr_im, blur_filter)

    return pyr, blur_filter


def _get_binomial_coefficients(size):
  """
  creates gaussian vector of the given size. size must be odd.
  :param size: odd integer
  :return: gaussian vector of the given size
  """
  binomial_coefficients = np.array([1, 1], dtype=np.float64)
  for _ in range(size - 2):
    binomial_coefficients = np.convolve([1, 1], binomial_coefficients)
  binomial_coefficients = binomial_coefficients[np.newaxis, :]
  return binomial_coefficients