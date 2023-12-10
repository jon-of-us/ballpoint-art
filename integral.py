import numpy as np
import cv2
import matplotlib.pyplot as plt

"""will not be used in the first hole"""
relative_margin = 0.05
relative_blur_kernel_size = 0.05
iterations = 300
cut_off_creep = 0.1


def convert(image: np.ndarray):
    normalized = image / 255
    integral = np.zeros(image.shape, np.float32)
    hole = np.zeros(image.shape, np.int8)
    margin = int(relative_margin * np.min(image.shape))
    hole[margin:-margin, margin:-margin] = 1

    for i in range(iterations):
        edge = normalized * (1 - hole)
        blur_kernel_size = int(relative_blur_kernel_size * np.min(image.shape))
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blurred = cv2.GaussianBlur(edge, (blur_kernel_size, blur_kernel_size), 0)
        creep_in = blurred * hole
        creep_in = creep_in > cut_off_creep
        hole -= creep_in
        integral += hole

    return integral
