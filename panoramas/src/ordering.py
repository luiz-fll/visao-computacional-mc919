import numpy as np


def build_connectivity_matrix(match_counts):
    """
    Cria uma matriz de conectividade a partir da quantidade
    de matches entre cada par de imagens.

    Parameters
    ----------
    match_counts : dict
        Dicionário no formato:
        {(i, j): número_de_matches}

    Returns
    -------
    matrix : numpy.ndarray
        Matriz simétrica de conectividade.
    """

    n = 0

    for i, j in match_counts:
        n = max(n, i + 1, j + 1)

    matrix = np.zeros((n, n), dtype=int)

    for (i, j), count in match_counts.items():
        matrix[i, j] = count
        matrix[j, i] = count

    return matrix


def build_graph(matrix, min_matches):
    """
    Cria um grafo a partir da matriz de conectividade.

    Duas imagens são consideradas vizinhas quando possuem
    pelo menos `min_matches` correspondências.
    """

    n = matrix.shape[0]

    graph = {
        i: []
        for i in range(n)
    }

    for i in range(n):
        for j in range(i + 1, n):

            if matrix[i, j] >= min_matches:
                graph[i].append(j)
                graph[j].append(i)

    return graph