"""
Ordenação das imagens e detecção de outliers com base na matriz de conectividade.

A força de conexão entre duas imagens é definida pelo número de matches mútuos
após o teste de Lowe.
"""
import numpy as np


def compute_image_statistics(connectivity_matrix):
    """
    Calcula estatísticas de conectividade para cada imagem.

    O score bilateral de uma aresta (i, j) é:
        W[i, j] / sqrt(median_i * median_j)

    onde median_i é a mediana das conexões da imagem i (excluindo a diagonal).

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz simétrica N×N com a força das conexões (número de matches).

    Returns
    -------
    dict
        Contém:
            - total_strength
            - median_strength
            - max_strength
            - relative_strength (max / mediana)
            - bilateral_scores (matriz)
            - bilateral_mean
            - bilateral_max
    """
    n = len(connectivity_matrix)

    total_strength = np.sum(connectivity_matrix, axis=1)

    median_strength = np.zeros(n)
    for i in range(n):
        values = np.delete(connectivity_matrix[i], i)
        median_strength[i] = np.median(values)

    max_strength = np.max(connectivity_matrix, axis=1)

    relative_strength = np.zeros(n)
    valid = median_strength > 0
    relative_strength[valid] = max_strength[valid] / median_strength[valid]

    # Matriz de scores bilaterais
    bilateral_scores = np.zeros_like(connectivity_matrix, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            denom = np.sqrt(median_strength[i] * median_strength[j])
            score = connectivity_matrix[i, j] / denom if denom > 0 else 0.0
            bilateral_scores[i, j] = score
            bilateral_scores[j, i] = score

    bilateral_mean = np.zeros(n)
    bilateral_max = np.zeros(n)
    for i in range(n):
        values = np.delete(bilateral_scores[i], i)
        bilateral_mean[i] = np.mean(values)
        bilateral_max[i] = np.max(values)

    return {
        "total_strength": total_strength,
        "median_strength": median_strength,
        "max_strength": max_strength,
        "relative_strength": relative_strength,
        "bilateral_scores": bilateral_scores,
        "bilateral_mean": bilateral_mean,
        "bilateral_max": bilateral_max,
    }


def detect_outliers(connectivity_matrix, threshold=0.5):
    """
    Detecta imagens candidatas a outliers com base no score bilateral médio.

    Uma imagem é considerada outlier quando:
        bilateral_mean < threshold

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade.
    threshold : float
        Limiar do score bilateral médio. Valores menores são mais permissivos.

    Returns
    -------
    list[int]
        Índices das imagens consideradas outliers.
    """
    statistics = compute_image_statistics(connectivity_matrix)
    scores = statistics["bilateral_mean"]
    return [i for i, score in enumerate(scores) if score < threshold]


def maximum_weight_hamiltonian_path(connectivity_matrix):
    """
    Encontra o caminho Hamiltoniano de peso máximo (cada imagem aparece exatamente uma vez).

    O peso de uma transição i → j é W[i, j].
    Utiliza programação dinâmica no estilo Held-Karp.

    Complexidade: O(n² · 2ⁿ) — viável para os datasets pequenos típicos desta atividade.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz simétrica de pesos.

    Returns
    -------
    path : list[int]
        Ordem das imagens que maximiza a soma dos pesos das arestas.
    """
    n = len(connectivity_matrix)
    n_masks = 1 << n

    # dp[mask, j] = melhor score de um caminho que visita exatamente o conjunto 'mask'
    # e termina na imagem j.
    dp = np.full((n_masks, n), -np.inf)
    parent = np.full((n_masks, n), -1, dtype=int)

    # Caminhos de uma única imagem têm score zero
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
                candidate = dp[previous_mask, k] + connectivity_matrix[k, j]
                if candidate > best_score:
                    best_score = candidate
                    best_prev = k

            dp[mask, j] = best_score
            parent[mask, j] = best_prev

    full_mask = n_masks - 1
    end = int(np.argmax(dp[full_mask]))

    # Reconstrução do caminho
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


def infer_order(matches_matrix, outlier_threshold=0.5):
    """
    Detecta outliers e ordena as imagens restantes pelo caminho Hamiltoniano de peso máximo.

    Parameters
    ----------
    matches_matrix : np.ndarray (dtype=object)
        Matriz de matches (normalmente a lowe_matrix). A força de conexão é o
        comprimento da lista de matches.
    outlier_threshold : float
        Limiar do score bilateral médio para detecção de outliers.

    Returns
    -------
    order : list[int]
        Ordem das imagens (apenas inliers).
    connectivity_matrix : np.ndarray
        Matriz de conectividade completa (número de matches).
    filtered_matrix : np.ndarray
        Submatriz de conectividade apenas com os inliers.
    """
    connectivity_matrix = np.vectorize(len)(matches_matrix)
    outliers = detect_outliers(connectivity_matrix, threshold=outlier_threshold)

    inliers = [i for i in range(len(connectivity_matrix)) if i not in outliers]

    if len(inliers) == 0:
        raise ValueError("O detector classificou todas as imagens como outliers.")
    if len(inliers) == 1:
        return inliers, connectivity_matrix, connectivity_matrix[np.ix_(inliers, inliers)]

    filtered_matrix = connectivity_matrix[np.ix_(inliers, inliers)]
    local_order = maximum_weight_hamiltonian_path(filtered_matrix)

    # Converte índices locais de volta para os índices originais
    order = [inliers[i] for i in local_order]
    return order, connectivity_matrix, filtered_matrix