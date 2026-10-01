import cv2
import numpy as np

def match_descriptors(descriptors1, descriptors2, method="sift"):
    """Faz matching direcional usando KNN (k=2)."""
    if descriptors1 is None or descriptors2 is None:
        return []

    if method.lower() == "sift":
        norm = cv2.NORM_L2
    elif method.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("Método deve ser 'sift' ou 'orb'.")

    matcher = cv2.BFMatcher(norm)

    return matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )


def lowe_ratio_test(matches, ratio=0.75):
    """Aplica o teste de razão de Lowe."""
    lowe_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        best, second = pair

        if best.distance < ratio * second.distance:
            lowe_matches.append(best)

    return lowe_matches


def mutual_matches(matches_12, matches_21):
    """
    Mantém somente correspondências que são consistentes nos dois sentidos.

    matches_12: matches de imagem 1 -> imagem 2.
    matches_21: matches de imagem 2 -> imagem 1.

    O retorno está na direção 1 -> 2.
    """
    reverse_pairs = {
        (match.trainIdx, match.queryIdx)
        for match in matches_21
    }

    return [
        match
        for match in matches_12
        if (match.queryIdx, match.trainIdx) in reverse_pairs
    ]


def match_all_images(descriptors, method="sift", ratio=0.75):
    """
    Faz matching bilateral entre todos os pares de imagens.

    Retorna duas matrizes NxN de dtype=object:

        matches_matrix[i, j]
            matches KNN de i -> j.

        lowe_matrix[i, j]
            matches que passaram no Lowe e também são mútuos.

    A diagonal fica vazia.
    As duas metades da matriz preservam a direção dos matches.
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
            # Matching nos dois sentidos.
            matches_ij = match_descriptors(
                descriptors[i],
                descriptors[j],
                method=method
            )
            matches_ji = match_descriptors(
                descriptors[j],
                descriptors[i],
                method=method
            )

            # Lowe nos dois sentidos.
            lowe_ij = lowe_ratio_test(
                matches_ij,
                ratio=ratio
            )
            lowe_ji = lowe_ratio_test(
                matches_ji,
                ratio=ratio
            )

            # Mantém somente matches que aparecem nos dois sentidos.
            mutual_ij = mutual_matches(lowe_ij, lowe_ji)
            mutual_ji = mutual_matches(lowe_ji, lowe_ij)

            matches_matrix[i, j] = [neighbor[0] for neighbor in matches_ij]
            matches_matrix[j, i] = [neighbor[0] for neighbor in matches_ji]

            lowe_matrix[i, j] = mutual_ij
            lowe_matrix[j, i] = mutual_ji

    return matches_matrix, lowe_matrix
