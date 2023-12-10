import numpy as np
import cv2

level_sets = 9
blur_kernel_size = 7


def convert(image: np.ndarray, scale=0):
    # image = cv2.bilateralFilter(image, 20, 1000, 0)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    image = np.array(np.floor(image / 255 * level_sets) * 255 / level_sets, np.uint8)
    image = cv2.medianBlur(image, blur_kernel_size)
    return image
