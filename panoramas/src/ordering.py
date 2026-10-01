import numpy as np

def lowe_to_count_matrix(lowe_matrix):
    """
    Converte uma matriz de listas de DMatch em uma matriz simétrica
    contendo a quantidade de matches bilaterais de cada par.
    """
    n = lowe_matrix.shape[0]
    counts = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            count_ij = len(lowe_matrix[i, j])
            count_ji = len(lowe_matrix[j, i])

            # Em princípio devem ser iguais. Usamos o menor valor para
            # evitar assimetrias acidentais na estrutura recebida.
            count = min(count_ij, count_ji)

            counts[i, j] = count
            counts[j, i] = count

    return counts


def normalize_connectivity(counts):
    """
    Normaliza a conectividade em relação à melhor conexão de cada imagem.

    Para cada imagem i:

        relative[i,j] = counts[i,j] / max_j(counts[i,j])

    Como a normalização por linha não é simétrica, combinamos as duas
    direções pela média:

        C[i,j] = 0.5 * (relative[i,j] + relative[j,i])

    Assim, C fica entre 0 e 1 e C[i,j] == C[j,i].
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.shape[0]

    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("A matriz de conectividade deve ser quadrada.")

    best = counts.max(axis=1)

    relative = np.zeros_like(counts, dtype=float)

    for i in range(n):
        if best[i] > 0:
            relative[i] = counts[i] / best[i]

    connectivity = 0.5 * (relative + relative.T)
    np.fill_diagonal(connectivity, 0.0)

    return connectivity, best


def detect_intruders(counts, intruder_ratio=0.25):
    """
    Detecta imagens cuja melhor conexão é muito fraca em relação
    à força típica das imagens do conjunto.

    A força de uma imagem é seu maior número de matches bilaterais.
    Normalizamos essa força pela mediana das forças do conjunto.

    Retorna:
        valid_mask: True para imagens mantidas.
        strength: força normalizada de cada imagem.
    """
    counts = np.asarray(counts, dtype=float)
    best = counts.max(axis=1)

    positive = best[best > 0]

    if len(positive) == 0:
        return np.zeros(len(best), dtype=bool), np.zeros(len(best))

    reference = np.median(positive)
    strength = best / reference

    valid_mask = strength >= intruder_ratio

    return valid_mask, strength


def maximum_spanning_tree(connectivity, valid_mask=None):
    """
    Calcula uma árvore geradora máxima usando Kruskal.

    Retorna uma lista de arestas:
        [(i, j, peso), ...]

    Somente imagens válidas participam da árvore.
    """
    connectivity = np.asarray(connectivity, dtype=float)
    n = connectivity.shape[0]

    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return False

        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a

        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

        return True

    edges = []

    for i in range(n):
        if not valid_mask[i]:
            continue

        for j in range(i + 1, n):
            if not valid_mask[j]:
                continue

            weight = connectivity[i, j]

            if weight > 0:
                edges.append((weight, i, j))

    # Maior peso primeiro.
    edges.sort(reverse=True)

    tree = []

    for weight, i, j in edges:
        if union(i, j):
            tree.append((i, j, weight))

    return tree


def order_from_tree(tree, valid_mask):
    """
    Extrai a ordem de uma árvore que representa uma cadeia.

    Para um panorama linear, a árvore esperada tem exatamente dois
    vértices de grau 1 e todos os outros de grau 2.

    Se a árvore tiver ramificações, a função gera erro em vez de
    inventar uma ordem que não esteja representada pelo grafo.
    """
    valid_nodes = np.flatnonzero(valid_mask).tolist()

    if not valid_nodes:
        return []

    if len(valid_nodes) == 1:
        return valid_nodes

    graph = {node: [] for node in valid_nodes}

    for i, j, weight in tree:
        graph[i].append((j, weight))
        graph[j].append((i, weight))

    degrees = {node: len(neighbors) for node, neighbors in graph.items()}

    if any(degree > 2 for degree in degrees.values()):
        raise ValueError(
            "A MST possui ramificações; ela não representa uma cadeia linear."
        )

    endpoints = [
        node for node, degree in degrees.items()
        if degree == 1
    ]

    if len(endpoints) != 2:
        raise ValueError(
            "A MST não possui exatamente duas extremidades."
        )

    start = endpoints[0]
    order = [start]
    previous = None
    current = start

    while True:
        candidates = [
            node
            for node, _ in graph[current]
            if node != previous
        ]

        if not candidates:
            break

        next_node = candidates[0]
        order.append(next_node)
        previous, current = current, next_node

    if len(order) != len(valid_nodes):
        raise ValueError(
            "Não foi possível percorrer todas as imagens da MST."
        )

    return order


def infer_order(lowe_matrix, intruder_ratio=0.25):
    """
    Pipeline completo de ordenação:

        Lowe bilateral
            ↓
        contagem de matches
            ↓
        normalização da conectividade
            ↓
        rejeição de intrusas
            ↓
        MST
            ↓
        ordenação da cadeia

    Retorna:
        order
        counts
        connectivity
        valid_mask
        strength
        tree
    """
    counts = lowe_to_count_matrix(lowe_matrix)

    connectivity, _ = normalize_connectivity(counts)

    valid_mask, strength = detect_intruders(
        counts,
        intruder_ratio=intruder_ratio
    )

    tree = maximum_spanning_tree(
        connectivity,
        valid_mask=valid_mask
    )

    order = order_from_tree(
        tree,
        valid_mask
    )

    return (
        order,
        counts,
        connectivity,
        valid_mask,
        strength,
        tree,
    )
