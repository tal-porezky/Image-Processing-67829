# ----------- Imports -----------

import numpy as np
from imageio import imwrite, imread
import scipy.ndimage  # todo ?
from tensorflow.keras.layers import Conv2D, Activation, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from skimage.color import rgb2gray
import sol5_utils  # todo ?


# ----------- Helper functions -----------


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


# ----------- 3: Dataset handling  -----------

def _read_images_and_put_in_dict(filenames, cache, batch_size):
    """
    reads images and puts them in the dict, if needed.
    :param filenames: list of filenames of images
    :param cache: dict with keys as filenames and images as values
    :param batch_size:  The size of the batch of images for each iteration
    of Stochasic Gradient Descent.
    :return: list of the chosen filenames
    """
    chosen_filenames = np.random.choice(filenames, size=batch_size)
    for chosen_filename in chosen_filenames:
        if chosen_filename not in cache.keys():
            im = read_image(chosen_filename, 1)
            cache[chosen_filename] = im
    return chosen_filenames


def _get_random_patch_of_the_image(im, crop_size):
    """
    crops the image to size crop_size and returns it
    :param im: image to crop
    :param crop_size: A tuple (height, width) specifying the crop size of
    the patches to extract.
    :return: same image but cropped to size crop_size in random location.
    """
    y, x = im.shape
    rand_x = np.random.randint(x - crop_size[1])
    rand_y = np.random.randint(y - crop_size[0])
    patch = im[rand_y:rand_y + crop_size[0],
            rand_x:rand_x + crop_size[1]]
    return patch


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Creates a generator of corrupt and normal image.
    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration
    of Stochasic Gradient Descent.
    :param corruption_func: A function receiving a numpy's array
    representation of an image as a single argument, and returns a randomly
    corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of
    the patches to extract.
    :return: Pythons generator which outputs random tuples of the form (
    sorce_batch, target_batch), where each output variable is an array of
    shape (batch_size, height, width, 1)
    """
    cache = dict()
    while True:
        source_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        target_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        chosen_filenames = _read_images_and_put_in_dict(filenames,
                                                        cache,
                                                        batch_size)
        for filename_idx in range(len(chosen_filenames)):
            filename = chosen_filenames[filename_idx]
            im = cache[filename]
            patch = _get_random_patch_of_the_image(im, crop_size)
            corrupted_im = corruption_func(patch)
            source_batch[filename_idx, :, :, 0] = corrupted_im - 0.5
            target_batch[filename_idx, :, :, 0] = patch - 0.5
        yield source_batch, target_batch


# ----------- 4: Neural Network Model  -----------


def resblock(input_tensor, num_channels):
    """
    Takes as input symbolic input tensor and the number of channels for
    each of its convolutional layers, and returns the symbolic output
    tensor of the layer configuration described in the PDF.
    :param input_tensor: input tensor...
    :param num_channels: the number of channels for
    each of its convolutional layers
    :return: output tensor of the layer configuration described in the PDF.
    """
    tensor = Conv2D(filters=num_channels, kernel_size=3, padding='same')(
        input_tensor)
    tensor = Activation('relu')(tensor)
    tensor = Conv2D(filters=num_channels, kernel_size=3, padding='same')(
        tensor)
    tensor = Add()([tensor, input_tensor])
    output_tensor = Activation('relu')(tensor)
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Returns output of the whole neural network
    :param height: number of pixels in the height
    :param width: number of pixels in the width
    :param num_channels: number of res block in the network
    :param num_res_blocks  number of res block in the network
    :return: output
    """
    input_network = Input(shape=(height, width, 1))
    tensor = Conv2D(num_channels, (3, 3), padding='same')(input_network)
    tensor = Activation('relu')(tensor)
    for _ in range(num_res_blocks):
        tensor = resblock(tensor, num_channels)
    tensor = Conv2D(1, (3, 3), padding='same')(tensor)
    tensor = Add()([tensor, input_network])
    model = Model(inputs=input_network, outputs=tensor)
    return model


# ----------- 5: Training Networks for Image Restoration  -----------


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    """
    trains the model.
    :param model: a general neural network model for image restoration.
    :param images: a list of le paths pointing to image les. You should
    assume these paths are complete, and should append anything to them.
    :param corruption_func: same as described in section 3.
    :param batch_size: the size of the batch of examples for each iteration of
    SGD.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to
    test on after every epoch.
    :return: none
    """
    train_set_size = np.int(len(images) * 0.8)
    train_images = images[: train_set_size]
    validation_images = images[train_set_size :]
    train_set = load_dataset(train_images,
                             batch_size,
                             corruption_func,
                             model.input_shape[1: 3])
    validation_set = load_dataset(validation_images,
                                  batch_size,
                                  corruption_func,
                                  model.input_shape[1: 3])
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_set, steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs, validation_data=validation_set,
                        validation_steps=(num_valid_samples // batch_size),
                        use_multiprocessing=True)
    # todo


# ----------- 6: Image Restoration of Complete Images  -----------


def restore_image(corrupted_image, base_model):
    """
    using our model to restore big size images.
    :param corrupted_image: a grayscale image of shape (height, width) and
    with values in the [0; 1] range of type float64.
    :param base_model: a neural network trained to restore small patches.
    :return:
    """
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    updated_corrupted_image = corrupted_image.reshape(
        corrupted_image.shape[0], corrupted_image.shape[1], 1)
    im = new_model.predict(np.expand_dims(updated_corrupted_image - 0.5, axis=0),
                      batch_size=1)[0]
    return np.clip(np.squeeze((im + 0.5), axis=2), 0, 1).astype('float64')


# ----------- 7: Application to Image Denoising and Deblurring  -----------


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    adds gaussian noise to the image.
    :param image: a grayscale image with values in the [0; 1] range of type
    float64.
    :param min_sigma: a non-negative scalar value representing the minimal
    variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to
    min_sigma, representing the maximal variance of the gaussian distribution.
    :return: noised image
    """
    assert(max_sigma >= min_sigma)
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    noisy_im = np.floor((image + noise) * 255) / 255
    noisy_im = np.clip(noisy_im, 0, 1)
    return noisy_im


def _gaussian_for_learn_denosing_model(image):
    """
    adds gaussian noise to image with sigma [0, 0.2]
    :param image: image
    :return: function which return image with noise
    """
    return add_gaussian_noise(image, 0, 0.2)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    returns trained denoised model
    :param num_res_blocks: number of res block in the network
    :param quick_mode: true or false value
    :return: the model.
    """
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        train_model(model, sol5_utils.images_for_denoising(),
                    _gaussian_for_learn_denosing_model, 10, 3, 2, 30)
        return model
    train_model(model, sol5_utils.images_for_denoising(),
                _gaussian_for_learn_denosing_model, 100, 100, 5, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    adds motion blur at given angel to the image.
    :param image: image
    :param kernel_size: int
    :param angle: an angle in radians in the range [0, pi).
    :return: blurred image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return scipy.ndimage.filters.convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """

    :param image: a grayscale image with values in the [0; 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return:
    """
    kernel_size = np.random.choice(list_of_kernel_sizes)
    angle = np.random.uniform(0, np.pi)
    noisy_im = add_motion_blur(image, kernel_size, angle)
    noisy_im = np.floor(noisy_im * 255) / 255
    noisy_im = np.clip(noisy_im, 0, 1)
    return noisy_im


def _motion_blur_for_learn_deblurring_model(image):
    """
    return an specific motion blur to image
    :param image: a grayscale image with values in the [0; 1] range of type float64.
    :return: function which return image with blur
    """
    return random_motion_blur(image, [7])


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    creates model which fixes deblurring images.
    :param num_res_blocks: number of res blocks in the network
    :param quick_mode: smaller model that make it run quickly
    :return: the model.
    """
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        train_model(model, sol5_utils.images_for_deblurring(),
                    _motion_blur_for_learn_deblurring_model,
                    10, 3, 2, 30)
        return model
    train_model(model, sol5_utils.images_for_deblurring(),
                _motion_blur_for_learn_deblurring_model,
                100, 100, 10, 1000)
    return model
