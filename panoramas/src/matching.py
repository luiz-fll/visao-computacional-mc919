"""
Matching de descritores entre pares de imagens.

Inclui:
- Matching KNN (k=2)
- Teste de razão de Lowe
- Filtragem de matches mútuos (cross-check)
"""
import cv2
import numpy as np

def match_descriptors(descriptors1, descriptors2, method="sift"):
    """
    Realiza matching direcional usando KNN com k=2.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Descritores da imagem de origem.
    descriptors2 : np.ndarray
        Descritores da imagem de destino.
    method : str
        "sift" (L2) ou "orb" (Hamming).

    Returns
    -------
    list
        Lista de listas de cv2.DMatch (até 2 vizinhos por descritor).
        Retorna lista vazia se algum dos descritores for None.
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if method.lower() == "sift":
        norm = cv2.NORM_L2
    elif method.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("method deve ser 'sift' ou 'orb'")

    matcher = cv2.BFMatcher(norm)
    return matcher.knnMatch(descriptors1, descriptors2, k=2)


def lowe_ratio_test(matches, ratio=0.75):
    """
    Aplica o teste de razão de Lowe.

    Mantém apenas matches em que a distância do melhor vizinho
    é significativamente menor que a do segundo melhor.

    Parameters
    ----------
    matches : list
        Resultado de knnMatch (lista de listas de DMatch).
    ratio : float
        Limiar de razão (tipicamente 0.7 ~ 0.8).

    Returns
    -------
    list[cv2.DMatch]
        Matches que passaram no teste.
    """
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good_matches.append(best)
    return good_matches


def mutual_matches(matches_12, matches_21):
    """
    Mantém apenas correspondências consistentes nos dois sentidos (cross-check).

    Parameters
    ----------
    matches_12 : list[cv2.DMatch]
        Matches da imagem 1 → 2.
    matches_21 : list[cv2.DMatch]
        Matches da imagem 2 → 1.

    Returns
    -------
    list[cv2.DMatch]
        Matches mútuos na direção 1 → 2.
    """
    reverse_pairs = {(m.trainIdx, m.queryIdx) for m in matches_21}
    return [
        m for m in matches_12
        if (m.queryIdx, m.trainIdx) in reverse_pairs
    ]


def match_all_images(descriptors, method="sift", ratio=0.75):
    """
    Realiza matching bilateral entre todos os pares de imagens.

    Parameters
    ----------
    descriptors : list[np.ndarray]
        Lista de descritores de cada imagem.
    method : str
        "sift" ou "orb".
    ratio : float
        Limiar do teste de Lowe.

    Returns
    -------
    matches_matrix : np.ndarray (dtype=object)
        Matriz N×N. matches_matrix[i, j] contém os matches brutos (melhor vizinho)
        da imagem i → j. Diagonal vazia.
    lowe_matrix : np.ndarray (dtype=object)
        Matriz N×N. lowe_matrix[i, j] contém os matches que passaram no
        teste de Lowe **e** são mútuos. Diagonal vazia.
    """
    n = len(descriptors)
    matches_matrix = np.empty((n, n), dtype=object)
    lowe_matrix = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            matches_matrix[i, j] = []
            lowe_matrix[i, j] = []

    for i in range(n):
        for j in range(i + 1, n):
            # Matching nos dois sentidos
            matches_ij = match_descriptors(descriptors[i], descriptors[j], method=method)
            matches_ji = match_descriptors(descriptors[j], descriptors[i], method=method)

            # Teste de Lowe nos dois sentidos
            lowe_ij = lowe_ratio_test(matches_ij, ratio=ratio)
            lowe_ji = lowe_ratio_test(matches_ji, ratio=ratio)

            # Cross-check
            mutual_ij = mutual_matches(lowe_ij, lowe_ji)
            mutual_ji = mutual_matches(lowe_ji, lowe_ij)

            # Guarda o melhor vizinho bruto (para eventual visualização)
            matches_matrix[i, j] = [pair[0] for pair in matches_ij if len(pair) > 0]
            matches_matrix[j, i] = [pair[0] for pair in matches_ji if len(pair) > 0]

            lowe_matrix[i, j] = mutual_ij
            lowe_matrix[j, i] = mutual_ji

    return matches_matrix, lowe_matrix