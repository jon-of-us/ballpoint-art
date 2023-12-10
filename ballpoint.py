import numpy as np
import cv2


stripe_width = 2
level_sets = 10
blur_kernel_size = 7
value_scale = 1


def stripes(shape, value, stripe_width=stripe_width):
    angle = np.random.rand() * np.pi * 2
    # angle = np.pi / 2
    vector = np.array([np.cos(angle), np.sin(angle)])
    cutoff = np.cos(np.pi * (1 - value))
    period_scale = (1 - value) * np.pi / stripe_width
    out_array = np.meshgrid(
        np.arange(shape[1]),
        np.arange(shape[0]),
    )
    # array of coordinate tuples
    out_array = np.stack(out_array, axis=-1)
    # dot product of each coordinate tuple with the vector
    out_array = np.dot(out_array, vector * period_scale)
    out_array = np.cos(out_array)
    # make black stripes with stripe_width
    out_array = np.uint8((out_array <= cutoff) * 255)

    return out_array


def convert(image: np.ndarray, scale=0):
    normalized = image / 255
    rounded = np.uint8(np.floor(normalized * level_sets))
    rounded = cv2.medianBlur(rounded, blur_kernel_size)

    out_image = np.zeros(image.shape, np.float32)
    for i in range(level_sets):
        stripe_image = stripes(
            image.shape, (i / level_sets) * value_scale - value_scale + 1
        )
        out_image += stripe_image * (rounded == i)

    return out_image
