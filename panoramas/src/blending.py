"""
Técnicas de blending para panoramas.

Inclui:
- Feather blending (peso proporcional à distância até a borda da máscara)
- Optimal seam + feathering local (para pares de imagens)
- Blending sequencial de múltiplas imagens
"""
import cv2
import numpy as np

def feather_blending(images, masks):
    """
    Feather blending multi-imagem.

    Cada pixel recebe peso proporcional à sua distância até a borda
    da máscara da imagem correspondente. O resultado final é a média
    ponderada de todas as contribuições.

    Parameters
    ----------
    images : list[np.ndarray]
        Imagens já alinhadas (mesmo tamanho).
    masks : list[np.ndarray]
        Máscaras booleanas ou binárias correspondentes.

    Returns
    -------
    panorama : np.ndarray, dtype=uint8
    """
    height, width = images[0].shape[:2]

    panorama_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum   = np.zeros((height, width),    dtype=np.float32)

    for image, mask in zip(images, masks):
        mask_binary = (mask > 0).astype(np.uint8)

        # Distância de cada pixel até a borda da região válida
        weight = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5).astype(np.float32)
        valid = weight > 0

        panorama_sum[valid] += image[valid].astype(np.float32) * weight[valid, None]
        weight_sum[valid]   += weight[valid]

    panorama = np.zeros_like(panorama_sum, dtype=np.float32)
    valid = weight_sum > 0
    panorama[valid] = panorama_sum[valid] / weight_sum[valid, None]

    return np.clip(panorama, 0, 255).astype(np.uint8)


def find_vertical_seam(cost):
    """
    Encontra a costura vertical de menor custo usando programação dinâmica.

    Parameters
    ----------
    cost : np.ndarray, shape (H, W)
        Matriz de custos (valores altos = regiões ruins para a costura).

    Returns
    -------
    seam : np.ndarray, shape (H,), dtype=int32
        Coordenada x da costura em cada linha y.
    """
    h, w = cost.shape
    dp = np.full((h, w), np.inf, dtype=np.float32)
    parent = np.zeros((h, w), dtype=np.int32)

    dp[0] = cost[0]

    for y in range(1, h):
        for x in range(w):
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            previous = dp[y - 1, x0:x1]
            best = np.argmin(previous)
            best_x = x0 + best
            dp[y, x] = cost[y, x] + dp[y - 1, best_x]
            parent[y, x] = best_x

    # Começa no menor custo da última linha e volta seguindo os pais
    x = int(np.argmin(dp[-1]))
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = x

    for y in range(h - 1, 0, -1):
        x = parent[y, x]
        seam[y - 1] = x

    return seam


def compensate_exposure_pair(image1, image2, mask1, mask2):
    """
    Ajusta a exposição de image2 para ficar semelhante à image1,
    usando apenas a região de sobreposição.

    Parameters
    ----------
    image1, image2 : np.ndarray
    mask1, mask2 : np.ndarray (booleanas ou binárias)

    Returns
    -------
    compensated : np.ndarray
        image2 com offset de exposição aplicado.
    """
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    overlap = mask1 & mask2

    if not np.any(overlap):
        return image2

    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)

    mean1 = np.mean(img1[overlap], axis=0)
    mean2 = np.mean(img2[overlap], axis=0)
    offset = mean1 - mean2

    compensated = np.clip(img2 + offset, 0, 255)
    return compensated.astype(np.uint8)


def optimal_seam_pair(image1, image2, mask1, mask2, feather_width=30):
    """
    Combina um par de imagens usando costura ótima + feathering local.

    Passos:
    1. Compensação de exposição na região de overlap.
    2. Cópia das regiões exclusivas.
    3. Cálculo da matriz de custo (|I1 − I2|) na região de overlap.
    4. Encontrar costura vertical de menor custo.
    5. Feathering linear em uma faixa de largura `feather_width` ao redor da costura.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Imagens já alinhadas.
    mask1, mask2 : np.ndarray
        Máscaras das regiões válidas.
    feather_width : int
        Largura da transição suave ao redor da costura.

    Returns
    -------
    result : np.ndarray
    """
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    overlap = mask1 & mask2

    result = np.zeros_like(image1)

    # Compensação de exposição
    image2 = compensate_exposure_pair(image1, image2, mask1, mask2)

    # Regiões exclusivas
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1
    result[only1] = image1[only1]
    result[only2] = image2[only2]

    if not np.any(overlap):
        return result

    # ROI da região de overlap
    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    image1_roi   = image1[y0:y1+1, x0:x1+1]
    image2_roi   = image2[y0:y1+1, x0:x1+1]
    overlap_roi  = overlap[y0:y1+1, x0:x1+1]

    # Matriz de custo
    gray1 = cv2.cvtColor(image1_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(image2_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cost = np.abs(gray1 - gray2)
    cost[~overlap_roi] = 1e6

    # Costura ótima
    seam = find_vertical_seam(cost)

    # Feathering ao redor da costura
    h_roi, w_roi = overlap_roi.shape
    feather_width = max(1, int(feather_width))

    for y in range(h_roi):
        global_y = y0 + y
        seam_x = x0 + seam[y]

        overlap_x = np.where(overlap[global_y])[0]
        if len(overlap_x) == 0:
            continue

        left  = max(overlap_x.min(), seam_x - feather_width)
        right = min(overlap_x.max(), seam_x + feather_width)

        for x in overlap_x:
            if x <= left:
                result[global_y, x] = image1[global_y, x]
            elif x >= right:
                result[global_y, x] = image2[global_y, x]
            else:
                alpha = (x - left) / float(right - left)
                blended = (
                    (1.0 - alpha) * image1[global_y, x].astype(np.float32) +
                    alpha         * image2[global_y, x].astype(np.float32)
                )
                result[global_y, x] = blended.astype(np.uint8)

    return result


def optimal_seam_blending(images, masks, feather_width=30):
    """
    Blending sequencial de múltiplas imagens usando optimal seam + feathering.

    Processo:
        panorama = imagem[0]
        para i = 1 … N-1:
            panorama = optimal_seam_pair(panorama, imagem[i], …)

    Parameters
    ----------
    images : list[np.ndarray]
        Imagens já alinhadas (mesmo tamanho).
    masks : list[np.ndarray]
        Máscaras correspondentes.
    feather_width : int
        Largura da transição suave.

    Returns
    -------
    panorama : np.ndarray
    """
    if len(images) == 0:
        raise ValueError("Nenhuma imagem fornecida.")
    if len(images) != len(masks):
        raise ValueError("O número de imagens e máscaras deve ser igual.")

    shape = images[0].shape
    for image, mask in zip(images, masks):
        if image.shape != shape:
            raise ValueError("Todas as imagens precisam ter o mesmo tamanho.")
        if mask.shape[:2] != shape[:2]:
            raise ValueError("Todas as máscaras precisam ter o mesmo tamanho.")

    panorama = images[0].copy()
    panorama_mask = masks[0] > 0

    for i in range(1, len(images)):
        print(f"Blending: imagem {i} + panorama")
        panorama = optimal_seam_pair(
            panorama, images[i],
            panorama_mask, masks[i],
            feather_width=feather_width,
        )
        panorama_mask = panorama_mask | (masks[i] > 0)

    return panorama