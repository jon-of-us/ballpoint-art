import numpy as np

"""between 0 and 1, 0 is no high pass, 1 is all high pass"""
lowpass_keep = (0.015, 0.07)
scale = 100


def lowpass_cutoff(shape, radius_percentage):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    distance_to_midpoint = x**2 + y**2
    return (distance_to_midpoint > (radius_percentage[0] * np.sqrt(2)) ** 2) * (
        distance_to_midpoint < (radius_percentage[1] * np.sqrt(2)) ** 2
    )


def convert(image: np.ndarray, scale=scale):
    shape = image.shape
    normalized = image / 255
    fourier = np.fft.fftshift(np.fft.fft2(normalized))
    lowpass_fourier = fourier.copy() * lowpass_cutoff(shape, lowpass_keep)
    highpass_fourier = fourier - lowpass_fourier
    # result_fourier = highpass_fourier
    result_fourier = lowpass_fourier
    result = np.abs(np.fft.ifft2(np.fft.fftshift(result_fourier)))
    avg = np.average(result.flat)
    result -= avg
    result = (np.tanh(result * scale) / 2 + 0.5) * 255
    return result
