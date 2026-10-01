import cv2
import numpy as np

def build_global_homographies(order, homographies, reference_position):
    """
    Constrói uma homografia global para cada imagem.

    G[i] transforma pontos da imagem i
    para o sistema de coordenadas da imagem de referência.
    """
    n = len(order)
    global_H = [None] * n
    global_H[reference_position] = np.eye(3)

    for k in range(reference_position - 1, -1, -1):
        H = homographies[k]

        # H: order[k] -> order[k+1]
        global_H[k] = global_H[k + 1] @ H

    for k in range(reference_position, n - 1):

        H = homographies[k]

        # H: order[k] -> order[k+1]
        # Precisamos do inverso para voltar
        # da imagem seguinte para a referência.
        H_inv = np.linalg.inv(H)

        global_H[k + 1] = global_H[k] @ H_inv

    return global_H

def project_points(H, points):
    """
    Projeta uma lista de pontos de acordo com a homografia H
    """
    points = np.asarray(points, dtype=np.float64)

    # Coordenadas homogêneas
    points_h = np.hstack([
        points,
        np.ones((len(points), 1))
    ])

    projected_h = points_h @ H.T

    w = projected_h[:, 2]
    projected = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64
    )

    valid = np.abs(w) > 1e-10

    projected[valid] = (projected_h[valid, :2] / w[valid, None])

    return projected

def get_image_corners(image):
    """
    Retorna os cantos de uma imagem como uma lista de pontos
    """
    h, w = image.shape[:2]

    return np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float64)

def to_cylindrical_coordinates(points, focal_length, cx, cy):
    """
    Converte uma lista de pontos para coordenadas cilíndricas
    """
    x = points[:, 0]
    y = points[:, 1]

    theta = np.arctan2(x - cx, focal_length)

    y_cyl = ((y - cy) * focal_length / np.sqrt((x - cx) ** 2 + focal_length ** 2))

    return theta, y_cyl

def cylindrical_bounds(images, order,global_homographies, focal_length):
    """
    Determina os limites da superfície cilíndrica
    """
    reference_image = images[order[len(order) // 2]]

    h, w = reference_image.shape[:2]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    all_theta = []
    all_y = []

    for position, image_index in enumerate(order):
        image = images[image_index]
        corners = get_image_corners(image)
        H = global_homographies[position]

        # Cantos da imagem transformados
        # para o sistema da referência.\
        corners_reference = project_points(H, corners)

        theta, y_cyl = to_cylindrical_coordinates(corners_reference, focal_length, cx, cy)

        all_theta.extend(theta)
        all_y.extend(y_cyl)

    theta_min = min(all_theta)
    theta_max = max(all_theta)

    y_min = min(all_y)
    y_max = max(all_y)

    return (theta_min, theta_max, y_min, y_max)

def cylindrical_inverse_map(canvas_width, canvas_height, theta_min, y_min, focal_length, cx, cy):
    """
    Mapeamento inverso em um canvas cilíndrico.

    Retorna 2 matrizes, x e y, que fornecem o mapa de posições de cada pixel do canvas em relação
    ao sistema de coordenadas da imagem de referência.

    x[i, j] = a
    y[i, j] = b
    canvas[i, j] = imagem[a, b]
    """
    u = np.arange(canvas_width, dtype=np.float64)
    v = np.arange(canvas_height,dtype=np.float64)

    U, V = np.meshgrid(u, v)

    # pixels do canvas -> coordenadas cilíndricas
    # theta determinado pro U, y determinado por V
    theta = (theta_min + U / focal_length)
    y_cyl = (y_min + V)

    # Mapeamento inverso
    x = (focal_length * np.tan(theta)) + cx
    y = (cy + y_cyl * np.sqrt(x ** 2 + focal_length ** 2) / focal_length)

    return x, y

def create_image_mask(map_x, map_y, image_shape):
    """
    Cria uma máscara indicando quais posições
    do canvas correspondem a pixels válidos
    da imagem.
    """
    h, w = image_shape[:2]

    mask = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0)
        & (map_x < w)
        & (map_y >= 0)
        & (map_y < h)
    )

    return mask

def average_blending(image1, image2, mask1, mask2):
    """
    Realiza mescla das imagens através da média da intensidade dos pixels
    """
    weight1 = mask1.astype(np.float32)
    weight2 = mask2.astype(np.float32)

    total_weight = (weight1 + weight2)

    blend = (
        image1.astype(np.float32)
        * weight1[..., None]
        +
        image2.astype(np.float32)
        * weight2[..., None]
    )

    valid = total_weight > 0

    blend[valid] /= (total_weight[valid, None])

    blend = np.clip(blend, 0, 255).astype(np.uint8)

    return blend

def cylindrical_stitching(images, order, homographies, focal_length, scale=1.0):
    """
    Cria um panorama cilíndrico utilizando múltiplas imagens.

    Parameters
    ----------
    images : list[np.ndarray]
        Imagens na ordem original/embaralhada.

    order : list[int]
        Ordem correta das imagens.

        Exemplo:
            [4, 5, 2, 6, 7, 3]

    homographies : list[np.ndarray]
        Homografias entre imagens consecutivas de order.

        homographies[0]:
            order[0] -> order[1]

        homographies[1]:
            order[1] -> order[2]

        ...

    focal_length : float
        Distância focal em pixels.

    scale : float
        Fator de escala aplicado às imagens.

    reference_position : int, optional
        Posição da imagem de referência dentro de order.
        Se None, utiliza a imagem central.

    Returns
    -------
    panorama : np.ndarray
        Panorama final.

    info : dict
        Informações da composição.
    """
    # Redimensionamento
    if scale != 1.0:

        images = [
            cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            for image in images
        ]

    f = focal_length * scale

    # Referência
    reference_position = len(order) // 2
    reference_index = order[reference_position]
    reference = images[reference_index]

    h, w = reference.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Homografias globais
    global_homographies = build_global_homographies(order, homographies, reference_position)

    # Limites do cilindro
    (theta_min, theta_max, y_min, y_max) = cylindrical_bounds(images, order, global_homographies, f)

    # Dimensões do canvas
    canvas_width = int(np.ceil((theta_max - theta_min) * f))
    canvas_height = int(np.ceil(y_max - y_min))

    print("Canvas:", canvas_width, "x", canvas_height)

    # Mapeamento canvas -> referência
    map_x_reference, map_y_reference = (
        cylindrical_inverse_map(
            canvas_width,
            canvas_height,
            theta_min,
            y_min,
            f,
            cx,
            cy
        )
    )

    points_reference = np.stack([
        map_x_reference.ravel(),
        map_y_reference.ravel()
    ], axis=1)

    # Projetar cada imagem no canvas
    warped_images = []
    masks = []
    for position, image_index in enumerate(order):
        image = images[image_index]
        H_global = global_homographies[position]

        print(f"Projetando imagem {image_index} (posição {position})")

        # Referência -> imagem atual
        points_image = project_points(np.linalg.inv(H_global), points_reference)

        map_x = points_image[:, 0].reshape(canvas_height, canvas_width)

        map_y = points_image[:, 1].reshape(canvas_height, canvas_width)

        # Máscara
        mask = create_image_mask(map_x, map_y, image.shape)

        # Remap
        warped = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        warped_images.append(warped)
        masks.append(mask)

    # Informações
    info = {
        "reference_position": reference_position,
        "reference_index": reference_index,
        "global_homographies": global_homographies,
        "theta_min_degrees": np.degrees(theta_min),
        "theta_max_degrees": np.degrees(theta_max),
        "y_min": y_min,
        "y_max": y_max,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
    }

    return warped_images, masks, info