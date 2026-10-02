import cv2
import numpy as np

def feather_blending(images, masks):
    """
    Feather blending.

    Mistura as imagens usando pesos proporcionais à distância
    de cada pixel até a borda da sua máscara.
    """

    height, width = images[0].shape[:2]

    panorama_sum = np.zeros(
        (height, width, 3),
        dtype=np.float32
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for image, mask in zip(images, masks):

        mask_binary = (mask > 0).astype(np.uint8)

        # Distância de cada pixel até a borda da região válida
        weight = cv2.distanceTransform(
            mask_binary,
            cv2.DIST_L2,
            5
        ).astype(np.float32)

        valid = weight > 0

        panorama_sum[valid] += (
            image[valid].astype(np.float32)
            * weight[valid, None]
        )

        weight_sum[valid] += weight[valid]

    panorama = np.zeros_like(
        panorama_sum,
        dtype=np.float32
    )

    valid = weight_sum > 0

    panorama[valid] = (
        panorama_sum[valid]
        / weight_sum[valid, None]
    )

    return np.clip(
        panorama,
        0,
        255
    ).astype(np.uint8)


def find_vertical_seam(cost):
    """
    Encontra uma costura vertical de menor custo
    usando programação dinâmica.
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

    # Começa no menor custo da última linha
    x = int(np.argmin(dp[-1]))

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = x

    # Volta seguindo os pais
    for y in range(h - 1, 0, -1):
        x = parent[y, x]
        seam[y - 1] = x

    return seam

def compensate_exposure_pair(image1, image2, mask1, mask2):
    """
    Ajusta a exposição da image2 para ficar semelhante à image1
    usando somente a região de sobreposição.
    """

    mask1 = mask1 > 0
    mask2 = mask2 > 0

    overlap = mask1 & mask2

    if not np.any(overlap):
        return image2

    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)

    # Médias de cada canal na região de sobreposição
    mean1 = np.mean(img1[overlap], axis=0)
    mean2 = np.mean(img2[overlap], axis=0)

    # Diferença de exposição
    offset = mean1 - mean2

    compensated = img2 + offset

    compensated = np.clip(
        compensated,
        0,
        255
    )

    return compensated.astype(np.uint8)

def optimal_seam_pair(
    image1,
    image2,
    mask1,
    mask2,
    feather_width=30
):

    mask1 = mask1 > 0
    mask2 = mask2 > 0

    overlap = mask1 & mask2

    result = np.zeros_like(image1)

    # --------------------------------------------------
    # Compensação de exposição
    # --------------------------------------------------

    image2 = compensate_exposure_pair(
        image1,
        image2,
        mask1,
        mask2
    )

    # --------------------------------------------------
    # Regiões exclusivas
    # --------------------------------------------------

    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1

    result[only1] = image1[only1]
    result[only2] = image2[only2]

    if not np.any(overlap):
        return result

    # --------------------------------------------------
    # Região de overlap
    # --------------------------------------------------

    ys, xs = np.where(overlap)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    image1_roi = image1[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    image2_roi = image2[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    overlap_roi = overlap[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    # --------------------------------------------------
    # Custo da costura
    # --------------------------------------------------

    gray1 = cv2.cvtColor(
        image1_roi,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    gray2 = cv2.cvtColor(
        image2_roi,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    cost = np.abs(gray1 - gray2)

    cost[~overlap_roi] = 1e6

    # --------------------------------------------------
    # Optimal seam
    # --------------------------------------------------

    seam = find_vertical_seam(cost)

    # --------------------------------------------------
    # Feather
    # --------------------------------------------------

    h_roi, w_roi = overlap_roi.shape

    feather_width = max(
        1,
        int(feather_width)
    )

    for y in range(h_roi):

        global_y = y0 + y

        seam_x = x0 + seam[y]

        overlap_x = np.where(
            overlap[global_y]
        )[0]

        if len(overlap_x) == 0:
            continue

        left = max(
            overlap_x.min(),
            seam_x - feather_width
        )

        right = min(
            overlap_x.max(),
            seam_x + feather_width
        )

        for x in overlap_x:

            if x <= left:

                result[global_y, x] = image1[
                    global_y, x
                ]

            elif x >= right:

                result[global_y, x] = image2[
                    global_y, x
                ]

            else:

                alpha = (
                    x - left
                ) / float(right - left)

                result[global_y, x] = (
                    (1.0 - alpha)
                    * image1[global_y, x].astype(np.float32)
                    +
                    alpha
                    * image2[global_y, x].astype(np.float32)
                ).astype(np.uint8)

    return result


def optimal_seam_blending(
    images,
    masks,
    feather_width=30
):
    """
    Faz o blending sequencial:

        imagem 1 + imagem 2
        resultado + imagem 3
        resultado + imagem 4
        ...

    usando optimal seam + feathering.
    """

    if len(images) == 0:
        raise ValueError("Nenhuma imagem fornecida.")

    if len(images) != len(masks):
        raise ValueError(
            "O número de imagens e máscaras deve ser igual."
        )

    # Todas as imagens precisam estar na mesma tela
    shape = images[0].shape

    for image, mask in zip(images, masks):

        if image.shape != shape:
            raise ValueError(
                "Todas as imagens precisam ter o mesmo tamanho."
            )

        if mask.shape[:2] != shape[:2]:
            raise ValueError(
                "Todas as máscaras precisam ter o mesmo tamanho."
            )

    panorama = images[0].copy()
    panorama_mask = masks[0] > 0

    for i in range(1, len(images)):

        print(
            f"Blending: imagem {i} + panorama"
        )

        panorama = optimal_seam_pair(
            panorama,
            images[i],
            panorama_mask,
            masks[i],
            feather_width=feather_width
        )

        panorama_mask = (
            panorama_mask |
            (masks[i] > 0)
        )

    return panorama

def find_mask_boundaries(masks):
    """
    Encontra as fronteiras das regiões válidas das máscaras.
    """

    boundaries = np.zeros_like(
        masks[0],
        dtype=np.uint8
    )

    for mask in masks:

        binary = (mask > 0).astype(np.uint8)

        # Erosão remove a camada externa.
        eroded = cv2.erode(
            binary,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        # O que existia antes e sumiu depois é a borda.
        boundary = binary - eroded

        boundaries |= boundary

    return boundaries

def remove_seam_lines(image, masks, thickness=3):
    """
    Remove pequenas linhas nas fronteiras das máscaras.

    Não modifica o fundo externo da panorâmica.
    """

    image = image.copy()

    # --------------------------------------------------
    # 1. União de todas as regiões válidas
    # --------------------------------------------------

    valid = np.zeros_like(
        masks[0],
        dtype=np.uint8
    )

    for mask in masks:
        valid |= (mask > 0).astype(np.uint8)

    # --------------------------------------------------
    # 2. Encontrar bordas das máscaras
    # --------------------------------------------------

    boundaries = find_mask_boundaries(masks)

    # --------------------------------------------------
    # 3. Expandir as bordas
    # --------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * thickness + 1, 2 * thickness + 1)
    )

    seam_region = cv2.dilate(
        boundaries,
        kernel,
        iterations=1
    )

    # --------------------------------------------------
    # 4. IMPORTANTE:
    # somente regiões dentro da panorâmica
    # --------------------------------------------------

    seam_region &= valid

    # --------------------------------------------------
    # 5. Inpainting somente nessas linhas
    # --------------------------------------------------

    inpaint_mask = (
        seam_region * 255
    ).astype(np.uint8)

    result = cv2.inpaint(
        image,
        inpaint_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    return result