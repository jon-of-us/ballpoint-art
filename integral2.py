import numpy as np
import cv2


"""will not be used in the first hole"""
iterations = 1000
# rel_min_creep = 0.01
# rel_max_creep = 0.03
min_creep = 5
max_creep = 15


def kernel_shape(max_creep):
    size = max_creep * 2 + 1
    return size, size


def kernel(min_creep, max_creep):
    x, y = np.meshgrid(
        np.arange(-max_creep, max_creep + 1), np.arange(-max_creep, max_creep + 1)
    )
    dist = np.sqrt(x**2 + y**2)
    res = 1 / (max_creep - min_creep) * (dist - min_creep)
    res[res < 0] = 0
    return res


def is_in_new_boundary(image, kernel, index):
    area = image[
        index[0] - kernel.shape[0] // 2 : index[0] + kernel.shape[0] // 2 + 1,
        index[1] - kernel.shape[1] // 2 : index[1] + kernel.shape[1] // 2 + 1,
    ]
    return np.any(area >= kernel)


def convert(image: np.ndarray):
    normalized = image / 255
    # top down
    # hole = np.zeros(image.shape, np.bool)
    # hole[1:, :] = True

    # eyes
    # hole = np.ones(image.shape, np.bool)
    # hole[210, 280] = False
    # hole[190, 205] = False

    # left to right
    hole = np.zeros(image.shape, np.bool)
    hole[:, 1:] = True

    # hole[529, 247 / 2] = False
    padded = np.pad(normalized, max_creep, mode="constant", constant_values=-1)
    hole = np.pad(hole, max_creep, mode="constant")
    ker = kernel(min_creep, max_creep)
    ker_shape = kernel_shape(max_creep)
    integral = np.zeros(image.shape, np.float32)
    for i in range(iterations):
        if not hole.any():
            break

        outside = padded.copy()
        outside[hole] = -1
        potential_new_area = hole & (cv2.GaussianBlur(outside, ker_shape, 0) > -1)

        indices = np.argwhere(potential_new_area)
        for index in indices:
            index = tuple(index)
            hole[index] &= not is_in_new_boundary(outside, ker, index)
        integral += hole[max_creep:-max_creep, max_creep:-max_creep]
    return integral
