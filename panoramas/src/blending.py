import numpy as np


def average_blending(images, masks):
    """
    Realiza blending por média das imagens.

    Parameters
    ----------
    images : list[np.ndarray]
        Imagens já projetadas no mesmo canvas.

    masks : list[np.ndarray]
        Máscaras booleanas indicando pixels válidos.

    Returns
    -------
    panorama : np.ndarray
        Panorama resultante.
    """
    height, width = images[0].shape[:2]

    panorama_sum = np.zeros((height, width, 3), dtype=np.float32)

    weight_sum = np.zeros((height, width), dtype=np.float32)

    for image, mask in zip(images, masks):
        weight = mask.astype(np.float32)

        panorama_sum += (image.astype(np.float32) * weight[..., None])

        weight_sum += weight

    panorama = np.zeros_like(panorama_sum)

    valid = weight_sum > 0

    panorama[valid] = (panorama_sum[valid] / weight_sum[valid, None])

    return np.clip(panorama, 0, 255).astype(np.uint8)

def median_blending(images, masks):
    stack = np.stack(images, axis=0).astype(np.float32)
    mask_stack = np.stack(masks, axis=0).astype(bool)

    # Repete a máscara para os 3 canais RGB/BGR.
    mask_stack = np.repeat(mask_stack[..., None], 3, axis=-1)

    # Onde a imagem não existe, usamos NaN.
    stack[~mask_stack] = np.nan

    panorama = np.nanmedian(stack, axis=0)

    # Pixels onde nenhuma imagem contribuiu.
    valid = np.any(mask_stack, axis=0)
    panorama[~valid] = 0

    return np.clip(panorama, 0, 255).astype(np.uint8)