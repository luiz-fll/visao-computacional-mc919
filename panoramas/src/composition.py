"""
Composição cilíndrica de panoramas.

Passos principais:
1. Construção de homografias globais a partir das homografias consecutivas.
2. Determinação dos limites da superfície cilíndrica.
3. Mapeamento inverso (canvas → sistema da imagem de referência → cada imagem).
4. Remapeamento (warping) de todas as imagens para o canvas comum.
"""

import cv2
import numpy as np


def build_global_homographies(order, homographies, reference_position):
    """
    Constrói uma homografia global para cada imagem.

    G[k] transforma pontos da imagem order[k] para o sistema de coordenadas
    da imagem de referência (order[reference_position]).

    Parameters
    ----------
    order : list[int]
        Ordem das imagens.
    homographies : list[np.ndarray]
        Homografias consecutivas: homographies[k] leva order[k] → order[k+1].
    reference_position : int
        Índice da imagem de referência dentro de `order`.

    Returns
    -------
    global_H : list[np.ndarray]
        Lista de homografias 3×3 (uma por posição em `order`).
    """
    n = len(order)
    global_H = [None] * n
    global_H[reference_position] = np.eye(3)

    # Para a esquerda da referência
    for k in range(reference_position - 1, -1, -1):
        # H: order[k] → order[k+1]
        global_H[k] = global_H[k + 1] @ homographies[k]

    # Para a direita da referência
    for k in range(reference_position, n - 1):
        # Precisamos do inverso: order[k+1] → order[k]
        H_inv = np.linalg.inv(homographies[k])
        global_H[k + 1] = global_H[k] @ H_inv

    return global_H


def project_points(H, points):
    """
    Projeta uma lista de pontos através da homografia H.

    Parameters
    ----------
    H : np.ndarray, shape (3, 3)
    points : np.ndarray, shape (N, 2)

    Returns
    -------
    projected : np.ndarray, shape (N, 2)
        Pontos inválidos recebem NaN.
    """
    points = np.asarray(points, dtype=np.float64)
    points_h = np.hstack([points, np.ones((len(points), 1))])
    projected_h = points_h @ H.T

    w = projected_h[:, 2]
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.abs(w) > 1e-10
    projected[valid] = projected_h[valid, :2] / w[valid, None]
    return projected


def get_image_corners(image):
    """
    Retorna os quatro cantos de uma imagem como array (4, 2).
    """
    h, w = image.shape[:2]
    return np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1]
    ], dtype=np.float64)


def to_cylindrical_coordinates(points, focal_length, cx, cy):
    """
    Converte pontos do plano da imagem de referência para coordenadas cilíndricas.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Pontos no sistema da imagem de referência.
    focal_length : float
        Distância focal em pixels.
    cx, cy : float
        Centro óptico da imagem de referência.

    Returns
    -------
    theta : np.ndarray
        Ângulo horizontal (radianos).
    y_cyl : np.ndarray
        Coordenada vertical cilíndrica.
    """
    x = points[:, 0]
    y = points[:, 1]

    theta = np.arctan2(x - cx, focal_length)
    y_cyl = (y - cy) * focal_length / np.sqrt((x - cx)**2 + focal_length**2)
    return theta, y_cyl


def cylindrical_bounds(images, order, global_homographies, focal_length):
    """
    Determina os limites da superfície cilíndrica a partir dos cantos de todas as imagens.

    Returns
    -------
    theta_min, theta_max, y_min, y_max : float
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

        # Cantos da imagem transformados para o sistema da referência
        corners_ref = project_points(H, corners)
        theta, y_cyl = to_cylindrical_coordinates(corners_ref, focal_length, cx, cy)

        all_theta.extend(theta)
        all_y.extend(y_cyl)

    return min(all_theta), max(all_theta), min(all_y), max(all_y)


def cylindrical_inverse_map(canvas_width, canvas_height, theta_min, y_min,
                            focal_length, cx, cy):
    """
    Mapeamento inverso do canvas cilíndrico para o sistema da imagem de referência.

    Para cada pixel (u, v) do canvas:
        theta = theta_min + u / f
        y_cyl = y_min + v
        x_ref = f * tan(theta) + cx
        y_ref = cy + y_cyl * sqrt(x_ref² + f²) / f

    Returns
    -------
    map_x, map_y : np.ndarray, shape (canvas_height, canvas_width)
        Coordenadas no sistema da imagem de referência.
    """
    u = np.arange(canvas_width, dtype=np.float64)
    v = np.arange(canvas_height, dtype=np.float64)
    U, V = np.meshgrid(u, v)

    theta = theta_min + U / focal_length
    y_cyl = y_min + V

    x = focal_length * np.tan(theta) + cx
    y = cy + y_cyl * np.sqrt(x**2 + focal_length**2) / focal_length
    return x, y


def create_image_mask(map_x, map_y, image_shape):
    """
    Cria máscara booleana indicando quais pixels do canvas caem dentro da imagem.
    """
    h, w = image_shape[:2]
    return (
        np.isfinite(map_x) & np.isfinite(map_y) &
        (map_x >= 0) & (map_x < w) &
        (map_y >= 0) & (map_y < h)
    )


def cylindrical_stitching(images, order, homographies, focal_length, verbose=True):
    """
    Cria um panorama cilíndrico a partir de múltiplas imagens.

    Parameters
    ----------
    images : list[np.ndarray]
        Imagens originais (na ordem dos índices 0..N-1).
    order : list[int]
        Ordem correta das imagens (ex.: [4, 5, 2, 6, 7, 3]).
    homographies : list[np.ndarray]
        Homografias consecutivas: homographies[k] leva order[k] → order[k+1].
    focal_length : float
        Distância focal em pixels.
    verbose : bool
        Se True, imprime o progresso da projeção.

    Returns
    -------
    warped_images : list[np.ndarray]
        Imagens projetadas no canvas cilíndrico.
    masks : list[np.ndarray]
        Máscaras booleanas correspondentes.
    info : dict
        Metadados da composição (dimensões do canvas, ângulos, etc.).
    """
    reference_position = len(order) // 2
    reference_index = order[reference_position]
    reference = images[reference_index]

    h, w = reference.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Homografias globais
    global_homographies = build_global_homographies(order, homographies, reference_position)

    # Limites do cilindro
    theta_min, theta_max, y_min, y_max = cylindrical_bounds(
        images, order, global_homographies, focal_length
    )

    # Dimensões do canvas
    canvas_width  = int(np.ceil((theta_max - theta_min) * focal_length))
    canvas_height = int(np.ceil(y_max - y_min))

    if verbose:
        print(f"[Composition] Canvas: {canvas_width} × {canvas_height}")
        print(f"[Composition] Referência: imagem {reference_index} (posição {reference_position})")

    # Mapeamento canvas → sistema da referência
    map_x_ref, map_y_ref = cylindrical_inverse_map(
        canvas_width, canvas_height,
        theta_min, y_min,
        focal_length, cx, cy
    )

    points_ref = np.stack([map_x_ref.ravel(), map_y_ref.ravel()], axis=1)

    # Projeta cada imagem no canvas
    warped_images = []
    masks = []

    for position, image_index in enumerate(order):
        image = images[image_index]
        H_global = global_homographies[position]

        if verbose:
            print(f"[Composition] Projetando imagem {image_index} (posição {position})")

        # Referência → imagem atual
        points_img = project_points(np.linalg.inv(H_global), points_ref)
        map_x = points_img[:, 0].reshape(canvas_height, canvas_width)
        map_y = points_img[:, 1].reshape(canvas_height, canvas_width)

        mask = create_image_mask(map_x, map_y, image.shape)

        warped = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        warped_images.append(warped)
        masks.append(mask)

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
