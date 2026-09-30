import cv2
import numpy as np


def compute_global_transforms(homographies, n_images, reference):
    """
    Calcula uma transformação global para cada imagem.

    Assume que homographies[(i, i+1)] transforma pontos
    da imagem i para o sistema de coordenadas da imagem i+1.
    """

    global_transforms = [None] * n_images
    global_transforms[reference] = np.eye(3, dtype=np.float64)

    # Imagens à esquerda da referência
    for i in range(reference - 1, -1, -1):
        H = homographies[(i, i + 1)]
        global_transforms[i] = global_transforms[i + 1] @ H

    # Imagens à direita da referência
    for i in range(reference + 1, n_images):
        H = homographies[(i - 1, i)]
        H_inv = np.linalg.inv(H)
        global_transforms[i] = global_transforms[i - 1] @ H_inv

    return global_transforms


def get_panorama_size(images, transforms):
    """
    Calcula o tamanho do canvas final do panorama.

    Retorna:
        width
        height
        translation
    """

    corners = []

    for image, H in zip(images, transforms):
        h, w = image.shape[:2]

        image_corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(
            image_corners,
            H
        )

        corners.append(transformed_corners.reshape(-1, 2))

    corners = np.vstack(corners)

    min_x = int(np.floor(corners[:, 0].min()))
    min_y = int(np.floor(corners[:, 1].min()))

    max_x = int(np.ceil(corners[:, 0].max()))
    max_y = int(np.ceil(corners[:, 1].max()))

    width = max_x - min_x
    height = max_y - min_y

    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float64)

    return width, height, translation


def compose_panorama(images, transforms):
    """
    Composição incremental do panorama.

    Apenas uma imagem é transformada por vez.
    Isso reduz bastante o consumo de memória.

    Retorna:
        panorama
        panorama_mask
    """

    width, height, translation = get_panorama_size(
        images,
        transforms
    )

    print(f"Canvas do panorama: {width} x {height}")

    # Acumulador em float32.
    # Possui 3 canais para RGB/BGR.
    accumulator = np.zeros(
        (height, width, 3),
        dtype=np.float32
    )

    # Soma dos pesos de cada pixel.
    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for i, (image, H) in enumerate(zip(images, transforms)):

        print(f"Processando imagem {i + 1}/{len(images)}...")

        H_global = translation @ H

        h, w = image.shape[:2]

        # Máscara da região válida da imagem.
        mask = np.ones(
            (h, w),
            dtype=np.uint8
        )

        # Warp da imagem atual.
        warped = cv2.warpPerspective(
            image,
            H_global,
            (width, height)
        )

        # Warp da máscara.
        warped_mask = cv2.warpPerspective(
            mask,
            H_global,
            (width, height)
        )

        valid = warped_mask > 0

        # Peso uniforme.
        accumulator[valid] += warped[valid].astype(np.float32)
        weight_sum[valid] += 1.0

        # Libera os arrays temporários antes da próxima imagem.
        del warped
        del warped_mask
        del mask

    valid = weight_sum > 0

    panorama = np.zeros_like(
        accumulator,
        dtype=np.uint8
    )

    panorama[valid] = (
        accumulator[valid] /
        weight_sum[valid, None]
    ).astype(np.uint8)

    panorama_mask = (
        weight_sum > 0
    ).astype(np.uint8) * 255

    del accumulator
    del weight_sum

    return panorama, panorama_mask


def feather_blend_panorama(images, transforms):
    """
    Composição incremental usando feather blending.

    Pixels próximos ao centro de cada imagem recebem
    peso maior do que pixels próximos às bordas.
    """

    width, height, translation = get_panorama_size(
        images,
        transforms
    )

    print(f"Canvas do panorama: {width} x {height}")

    accumulator = np.zeros(
        (height, width, 3),
        dtype=np.float32
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for i, (image, H) in enumerate(zip(images, transforms)):

        print(
            f"Blending imagem "
            f"{i + 1}/{len(images)}..."
        )

        H_global = translation @ H

        h, w = image.shape[:2]

        mask = np.ones(
            (h, w),
            dtype=np.uint8
        )

        warped = cv2.warpPerspective(
            image,
            H_global,
            (width, height)
        )

        warped_mask = cv2.warpPerspective(
            mask,
            H_global,
            (width, height)
        )

        # Distância até a borda da região válida.
        distance = cv2.distanceTransform(
            warped_mask,
            cv2.DIST_L2,
            5
        )

        # Evita pesos extremamente pequenos.
        distance += 1e-6

        valid = warped_mask > 0

        accumulator[valid] += (
            warped[valid].astype(np.float32)
            * distance[valid, None]
        )

        weight_sum[valid] += distance[valid]

        del warped
        del warped_mask
        del mask
        del distance

    valid = weight_sum > 0

    panorama = np.zeros_like(
        accumulator,
        dtype=np.uint8
    )

    panorama[valid] = (
        accumulator[valid]
        / weight_sum[valid, None]
    ).astype(np.uint8)

    panorama_mask = (
        weight_sum > 0
    ).astype(np.uint8) * 255

    del accumulator
    del weight_sum

    return panorama, panorama_mask


def largest_rectangle(binary):
    """
    Encontra o maior retângulo formado apenas por pixels 1.

    Parâmetros:
        binary: máscara 2D contendo 0 e 1.

    Retorna:
        (x, y, width, height)
    """

    rows, cols = binary.shape

    heights = [0] * cols

    best_area = 0
    best_rect = (0, 0, 0, 0)

    for y in range(rows):

        # Atualiza a altura de cada coluna.
        for x in range(cols):
            if binary[y, x]:
                heights[x] += 1
            else:
                heights[x] = 0

        # Maior retângulo no histograma desta linha.
        stack = []

        for x in range(cols + 1):

            current_height = (
                heights[x]
                if x < cols
                else 0
            )

            while stack and current_height < heights[stack[-1]]:

                height = heights[stack.pop()]

                if stack:
                    left = stack[-1] + 1
                else:
                    left = 0

                width = x - left

                area = width * height

                if area > best_area:
                    best_area = area

                    best_rect = (
                        left,
                        y - height + 1,
                        width,
                        height
                    )

            stack.append(x)

    return best_rect


def crop_panorama(panorama, mask):
    """
    Recorta o maior retângulo alinhado aos eixos
    completamente contido na região válida do panorama.

    Retorna:
        panorama recortado
    """

    # Converte a máscara para binária.
    binary = (mask > 0).astype(np.uint8)

    # Encontra a região geral que contém o panorama.
    x, y, w, h = cv2.boundingRect(binary)

    # Trabalha somente dentro dessa região.
    region = binary[
        y:y + h,
        x:x + w
    ]

    # Encontra o maior retângulo completamente válido.
    rx, ry, rw, rh = largest_rectangle(region)

    if rw == 0 or rh == 0:
        return panorama

    # Converte as coordenadas da região para
    # coordenadas do panorama original.
    x1 = x + rx
    y1 = y + ry
    x2 = x1 + rw
    y2 = y1 + rh

    return panorama[
        y1:y2,
        x1:x2
    ]