from os import walk
import numpy as np
from dijkstra_integral import convert
import cv2
import rawpy
import matplotlib.pyplot as plt

down_scale = 1


def files_in_folder(path):
    """yield relative path to all files in a directory"""
    for dirpath, dirnames, filenames in walk(path):
        for filename in filenames:
            yield dirpath + "/" + filename


def downscaled(image: np.ndarray, scale=down_scale):
    return cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))


def read_image(filepath):
    switch = filepath.split(".")[-1]
    if switch in ["jpg", "jpeg", "png"]:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    elif switch in ["raw", "dng"]:
        with rawpy.imread(filepath) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) / 256


def convert_all_to_video():
    print("converting all images to videos")
    for filepath in files_in_folder("./input"):
        # read image to numpy array in grayscale
        in_arr = read_image(filepath)
        in_arr = downscaled(in_arr)

        out_path = "./output/" + filepath.split("/")[-1].split(".")[0] + ".mp4"
        video = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.3,
            in_arr.shape[::-1],
        )

        for scale in range(0, 2500, 200):
            # out_arr = convert(in_arr, scale)
            out_image = cv2.cvtColor(
                np.uint8(convert(in_arr, scale)),
                cv2.COLOR_GRAY2RGB,
            )
            # out_image = cv2.resize(out_image, new_shape)
            # save image
            video.write(out_image)

        video.release()
        cv2.destroyAllWindows()


def convert_all_to_foto():
    print("converting all images to fotos")
    for filepath in files_in_folder("./input"):
        # read image to numpy array in grayscale
        in_arr = read_image(filepath)
        # in_arr = downscaled(in_arr)

        in_arr = downscaled(in_arr)
        out_arr = convert(in_arr)
        out_arr /= np.max(out_arr)
        out_arr *= 255
        out_arr = np.uint8(out_arr)
        out_path = "./output/" + filepath.split("/")[-1].split(".")[0] + ".png"

        # save image
        cv2.imwrite(out_path, out_arr, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def convert_all_to_array():
    print("converting all images to numpy arrays")
    for filepath in files_in_folder("./input"):
        # read image to numpy array in rgba
        in_arr = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        start_node_image = in_arr[:, :, 3]
        in_arr = cv2.cvtColor(in_arr, cv2.COLOR_RGBA2GRAY)

        out_arr = convert(in_arr, start_node_image)
        out_path = "./output_arrays/" + filepath.split("/")[-1].split(".")[0] + ".npy"

        # save image
        np.save(out_path, out_arr)


def show_contour_plot():
    for filepath in files_in_folder("./output_arrays"):
        out_arr = np.load(filepath)
        # dx, dy = np.gradient(out_arr)
        # plt.imshow(np.sqrt(dx**2 + dy**2))
        maximum = np.max(out_arr)
        plt.contour(out_arr, levels=250, colors="black", origin="image", linewidths=0.5)
        plt.axis("off")
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    # convert_all_to_foto()
    convert_all_to_array()
    show_contour_plot()
    # convert_all_to_video()
