import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("./input/arm3.jpg", cv2.IMREAD_GRAYSCALE)
width, height = image.shape
image = np.float32(image)
image = cv2.GaussianBlur(image, (11, 11), 0)
print(image[width // 2, :])
plt.plot(image[width // 2, :])
plt.show()
