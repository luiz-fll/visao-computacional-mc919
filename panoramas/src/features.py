"""
Detecção de pontos de interesse e descritores.

Suporta SIFT (mais preciso, mais lento) e ORB (mais rápido, binário).
"""
import cv2

def detect_sift(image):
    """
    Detecta keypoints e calcula descritores SIFT em uma imagem.

    Parameters
    ----------
    image : np.ndarray
        Imagem em escala de cinza ou BGR.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
    descriptors : np.ndarray ou None
        Matriz de descritores (N x 128). Pode ser None se nenhum ponto for encontrado.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def detect_orb(image):
    """
    Detecta keypoints e calcula descritores ORB em uma imagem.

    Parameters
    ----------
    image : np.ndarray
        Imagem em escala de cinza ou BGR.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
    descriptors : np.ndarray ou None
        Matriz de descritores binários (N x 32). Pode ser None se nenhum ponto for encontrado.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def draw_keypoints(images, method="sift"):
    """
    Detecta e desenha keypoints em uma lista de imagens.

    Útil para inspeção visual da qualidade da detecção.

    Parameters
    ----------
    images : list[np.ndarray]
        Lista de imagens BGR.
    method : str
        "sift" ou "orb".

    Returns
    -------
    list[dict]
        Cada dicionário contém:
            - "image": imagem com keypoints desenhados
            - "keypoints": lista de cv2.KeyPoint
            - "descriptors": matriz de descritores
    """
    if method.lower() == "sift":
        detector = detect_sift
    elif method.lower() == "orb":
        detector = detect_orb
    else:
        raise ValueError("method deve ser 'sift' ou 'orb'")

    results = []
    print(f"Detector: {method}")

    for image in images:
        keypoints, descriptors = detector(image)
        print(f"  Número de keypoints: {len(keypoints)}")
        if descriptors is not None:
            print(f"  Formato dos descritores: {descriptors.shape}")

        image_with_keypoints = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        results.append({
            "image": image_with_keypoints,
            "keypoints": keypoints,
            "descriptors": descriptors,
        })

    return results