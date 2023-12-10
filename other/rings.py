import numpy as np
import cv2

img = cv2.imread("./input/arm3.jpg", cv2.IMREAD_GRAYSCALE)

pixel_to_view = 76

# for i in range(255):
for pixel_to_view in range(74, 80):
    img = np.uint8(img)
    filtered_img = img * (img == pixel_to_view)

    # save image
    cv2.imwrite(f"./output/pixel{pixel_to_view}.jpg", filtered_img)
