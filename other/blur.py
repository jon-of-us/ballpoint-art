import numpy as np
import cv2

blur_strength = 0.1
denoise_kernel_size = 21
denoise_on = False
scale = 100000


def convert(image: np.ndarray, scale=scale):
    kernel_size = int(min(image.shape) // 2 * blur_strength) * 2 + 1
    normalized = image / 255

    if denoise_on:
        denoised = cv2.GaussianBlur(
            normalized, (denoise_kernel_size, denoise_kernel_size), 0
        )
        blurred = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), 0)
        denoised - blurred
    else:
        blurred = cv2.GaussianBlur(normalized, (kernel_size, kernel_size), 0)
        highpass = normalized - blurred

    scaled = (np.tanh(highpass / 255 * scale) + 1) / 2 * 255

    return scaled
