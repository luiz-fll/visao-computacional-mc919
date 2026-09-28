import cv2

def match_descriptors(descriptors1, descriptors2, method="sift"):
    """
    Encontra correspondências entre os descritores de duas imagens.

    Parameters
    ----------
    descriptors1 : numpy.ndarray
        Descritores da primeira imagem.
    descriptors2 : numpy.ndarray
        Descritores da segunda imagem.
    method : str
        Método utilizado para calcular a distância.
        "sift" utiliza distância L2.
        "orb" utiliza distância Hamming.

    Returns
    -------
    matches : list
        Lista de matches encontrados.
    """

    if method.lower() == "sift":
        norm = cv2.NORM_L2
    elif method.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("Método deve ser 'sift' ou 'orb'.")

    matcher = cv2.BFMatcher(norm)

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    return matches


def ratio_test(matches, ratio=0.75):
    """
    Aplica o Lowe Ratio Test aos matches.

    Parameters
    ----------
    matches : list
        Matches retornados por BFMatcher.knnMatch().
    ratio : float
        Threshold utilizado no Lowe Ratio Test.

    Returns
    -------
    good_matches : list
        Matches aprovados pelo teste.
    """

    good_matches = []

    for match_pair in matches:
        if len(match_pair) < 2:
            continue

        best_match, second_match = match_pair

        if best_match.distance < ratio * second_match.distance:
            good_matches.append(best_match)

    return good_matches