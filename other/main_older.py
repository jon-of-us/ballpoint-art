import PIL.Image
import numpy as np
import scipy
from os import walk

"""between 0 and 1, 0 is no high pass, 1 is all high pass"""
lowpass_keep = 0.9
scale = 10


def files_in_folder(path):
    """yield relative path to all files in a directory"""
    for dirpath, dirnames, filenames in walk(path):
        for filename in filenames:
            yield dirpath + "/" + filename


def exp_lowpass_grid(shape, strength):
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx, yy = np.meshgrid(y, x)
    return np.exp(-strength * (xx + yy))


def hyperbolic_lowpass_grid(shape, strength):
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx, yy = np.meshgrid(y, x)
    return 1 / (1 + strength * (xx + yy))


def cutoff_lowpass_grid(shape, percentage_to_keep):
    A = np.zeros(shape)
    A[
        : int(percentage_to_keep * shape[0]),
        : int(percentage_to_keep * shape[1]),
    ] = 1
    return A


def smur(im,n,p):
    x = im*(1-p)+(im-scipy.ndimage.gaussian_filter(im,n))*p
    return x/np.max(x)*255*2


def convert(image: np.ndarray):
    shape = image.shape
    fourier = np.fft.fft2(image)
    low_pass = fourier.copy() * cutoff_lowpass_grid(shape, lowpass_keep)
    high_pass = fourier - low_pass
    low_image = np.abs(np.fft.ifft2(low_pass))
    high_image = np.tanh(np.abs(np.fft.ifft2(high_pass)) / 255 * scale) * 255
    return high_image


for filepath in files_in_folder("./input"):
    in_arr = np.asarray(PIL.Image.open(filepath).convert("L"))
    out_arr = smur(in_arr,800,1)
    pil_image = PIL.Image.fromarray(np.uint8(out_arr))

    out_path = "./output/" + filepath.split("/")[-1]
    pil_image.save(out_path)
    break
