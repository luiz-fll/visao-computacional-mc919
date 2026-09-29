import cv2
import numpy as np


def blend_images(image1, image2, alpha=0.5):
    """
    Faz uma fusão simples entre duas imagens já alinhadas.
    """

    mask1 = np.any(image1 > 0, axis=2)
    mask2 = np.any(image2 > 0, axis=2)

    result = np.zeros_like(image1)

    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1
    overlap = mask1 & mask2

    result[only1] = image1[only1]
    result[only2] = image2[only2]

    result[overlap] = (
        alpha * image1[overlap] +
        (1 - alpha) * image2[overlap]
    ).astype(np.uint8)

    return result