"""
Ordenação automática das imagens e detecção de imagens intrusas.

A conectividade entre duas imagens é definida pelo número de matches
que passaram pelo teste de Lowe e pelo cross-check.

Para detectar imagens que não pertencem à sequência, calcula-se para
cada imagem a soma de suas conexões com todas as outras imagens.

Essa soma é normalizada pela mediana das somas do conjunto:

    r_i = T_i / median(T)

onde:
    T_i = soma das conexões da imagem i
    r_i = força relativa de conectividade

Uma imagem é considerada outlier quando sua força relativa é menor
que um limiar definido pelo usuário.

Depois da remoção dos outliers, a ordem das imagens é determinada
pelo caminho Hamiltoniano de peso máximo.
"""

import numpy as np


def compute_connectivity_statistics(connectivity_matrix):
    """
    Calcula estatísticas de conectividade das imagens.

    Para cada imagem i, calcula:

        T_i = sum_j W[i, j]

    onde W é a matriz de conectividade.

    Em seguida, normaliza cada total pela mediana dos totais:

        r_i = T_i / median(T)

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz N×N de conectividade. Cada elemento [i, j]
        representa o número de matches entre as imagens i e j.

    Returns
    -------
    dict
        Dicionário contendo:

        - total_strength : np.ndarray
            Soma das conexões de cada imagem.

        - median_total : float
            Mediana das somas de conectividade.

        - relative_strength : np.ndarray
            Total de cada imagem dividido pela mediana dos totais.
    """
    total_strength = np.sum(connectivity_matrix, axis=1)
    median_total = np.median(total_strength)

    if median_total > 0:
        relative_strength = total_strength / median_total
    else:
        relative_strength = np.zeros_like(
            total_strength,
            dtype=np.float64
        )

    return {
        "total_strength": total_strength,
        "median_total": median_total,
        "relative_strength": relative_strength,
    }


def detect_outliers(connectivity_matrix, outlier_threshold=0.3):
    """
    Detecta imagens que apresentam conectividade muito baixa.

    A detecção é baseada na razão entre a conectividade total
    da imagem e a mediana da conectividade total do conjunto:

        r_i = T_i / median(T)

    Uma imagem é considerada outlier quando:

        r_i < outlier_threshold

    O critério é inspirado na ideia de utilizar uma razão relativa
    para filtrar valores significativamente mais fracos, de forma
    análoga ao princípio do teste de razão de Lowe.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz N×N de conectividade.

    outlier_threshold : float, optional
        Limiar da razão de conectividade.
        O valor padrão é 0.3.

    Returns
    -------
    outliers : list[int]
        Índices das imagens consideradas outliers.

    statistics : dict
        Estatísticas calculadas por compute_connectivity_statistics().
    """
    statistics = compute_connectivity_statistics(connectivity_matrix)

    relative_strength = statistics["relative_strength"]

    outliers = [
        i
        for i, score in enumerate(relative_strength)
        if score < outlier_threshold
    ]

    return outliers, statistics


def maximum_weight_hamiltonian_path(connectivity_matrix):
    """
    Encontra o caminho Hamiltoniano de peso máximo.

    Cada imagem deve aparecer exatamente uma vez no caminho.

    O peso de uma transição entre duas imagens é dado pelo número
    de matches entre elas.

    O objetivo é encontrar:

        max sum_k W[pi_k, pi_{k+1}]

    onde pi representa uma possível ordenação das imagens.

    A implementação utiliza programação dinâmica no estilo Held-Karp.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade das imagens.

    Returns
    -------
    path : list[int]
        Ordem das imagens que maximiza a soma dos pesos das
        conexões consecutivas.
    """
    n = len(connectivity_matrix)
    n_masks = 1 << n

    # dp[mask, j] representa o melhor caminho que:
    # - visita exatamente as imagens de mask
    # - termina na imagem j
    dp = np.full((n_masks, n), -np.inf)
    parent = np.full((n_masks, n), -1, dtype=int)

    # Caminhos contendo apenas uma imagem têm peso zero.
    for j in range(n):
        dp[1 << j, j] = 0.0

    for mask in range(n_masks):
        for j in range(n):

            if not (mask & (1 << j)):
                continue

            previous_mask = mask ^ (1 << j)

            if previous_mask == 0:
                continue

            best_score = -np.inf
            best_prev = -1

            for k in range(n):

                if not (previous_mask & (1 << k)):
                    continue

                candidate = (
                    dp[previous_mask, k]
                    + connectivity_matrix[k, j]
                )

                if candidate > best_score:
                    best_score = candidate
                    best_prev = k

            dp[mask, j] = best_score
            parent[mask, j] = best_prev

    # Melhor final para o caminho completo.
    full_mask = n_masks - 1
    end = int(np.argmax(dp[full_mask]))

    # Reconstrução do caminho.
    path = []
    mask = full_mask
    current = end

    while current != -1:
        path.append(current)

        previous = parent[mask, current]

        mask ^= 1 << current
        current = previous

    path.reverse()

    return path


def infer_order(lowe_matrix, outlier_threshold=0.3):
    """
    Detecta imagens intrusas e determina automaticamente a ordem
    das imagens restantes.

    O processo possui duas etapas:

    1. Detecção de outliers
       A matriz de matches após Lowe + cross-check é convertida
       em uma matriz de conectividade. Para cada imagem calcula-se
       a conectividade total e sua razão em relação à mediana
       do conjunto.

    2. Ordenação
       Após remover os outliers, encontra-se o caminho Hamiltoniano
       de peso máximo sobre a matriz de conectividade restante.

    Parameters
    ----------
    lowe_matrix : np.ndarray
        Matriz N×N de listas de matches. Normalmente corresponde
        à matriz produzida por match_all_images(), contendo os
        matches que passaram pelo teste de Lowe e pelo cross-check.

    outlier_threshold : float, optional
        Limiar utilizado na detecção de outliers.

        Uma imagem é removida quando:

            total_i / median(total) < ratio_threshold

        O valor padrão é 0.3.

    Returns
    -------
    order : list[int]
        Ordem das imagens consideradas válidas.

    connectivity_matrix : np.ndarray
        Matriz N×N contendo o número de matches entre cada par.

    filtered_matrix : np.ndarray
        Submatriz da matriz de conectividade contendo somente
        as imagens consideradas válidas.

    statistics : dict
        Estatísticas utilizadas na detecção dos outliers:

        - total_strength
        - median_total
        - relative_strength

    outliers : list[int]
        Índices das imagens rejeitadas.
    """
    # Converte a matriz de listas de matches em uma matriz
    # numérica de conectividade.
    connectivity_matrix = np.vectorize(len)(lowe_matrix)

    # Detecta imagens intrusas.
    outliers, statistics = detect_outliers(
        connectivity_matrix,
        outlier_threshold=outlier_threshold
    )

    # Mantém somente as imagens consideradas válidas.
    inliers = [
        i
        for i in range(len(connectivity_matrix))
        if i not in outliers
    ]

    if len(inliers) == 0:
        raise ValueError(
            "O detector classificou todas as imagens como outliers."
        )

    if len(inliers) == 1:
        filtered_matrix = connectivity_matrix[
            np.ix_(inliers, inliers)
        ]

        return (
            inliers,
            connectivity_matrix,
            filtered_matrix,
            statistics
        )

    # Remove os outliers da matriz.
    filtered_matrix = connectivity_matrix[
        np.ix_(inliers, inliers)
    ]

    # Encontra a melhor sequência entre as imagens restantes.
    local_order = maximum_weight_hamiltonian_path(
        filtered_matrix
    )

    # Converte os índices locais novamente para os índices
    # originais das imagens.
    order = [
        inliers[i]
        for i in local_order
    ]

    return (
        order,
        connectivity_matrix,
        filtered_matrix,
        statistics
    )