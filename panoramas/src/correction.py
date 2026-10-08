"""
Correções na projeção cilíndrica, no plot da matriz e implementação de recorte retangular.
NÃO CONSTA NO RELATÓRIO.
"""
import numpy as np
import matplotlib.pyplot as plt

def cylindrical_inverse_map_CORRECTED(
    canvas_width, canvas_height,
    theta_min, y_min,
    focal_length, cx, cy
):
    u = np.arange(canvas_width, dtype=np.float64)
    v = np.arange(canvas_height, dtype=np.float64)
    U, V = np.meshgrid(u, v)

    theta = theta_min + U / focal_length
    y_cyl = y_min + V

    x = focal_length * np.tan(theta) + cx

    dx = x - cx

    # Corrige fórmula do remap, dx ao invés de x
    y = cy + y_cyl * np.sqrt(dx**2 + focal_length**2) / focal_length

    return x, y

def plot_matrix_CORRECTED(matrix, title="Matriz", order=None):

    if order is not None:
        order = np.asarray(order, dtype=int)

        if np.any(order < 0) or np.any(order >= matrix.shape[0]):
            raise ValueError("Há índices em order fora dos limites da matriz.")

        # Essa versão ordena as linhas e colunas, a versão antiga só mexia
        # nos rótulos das laterais
        matrix = matrix[np.ix_(order, order)]
        labels = order
    else:
        labels = np.arange(matrix.shape[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Número de matches")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value}",
                ha="center", va="center",
                color="white" if value < np.max(matrix) * 0.7 else "black"
            )

    ax.set_xlabel("Imagem")
    ax.set_ylabel("Imagem")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def crop_panorama(panorama, masks):
    """
    Recorta um panorama já composto usando suas máscaras de cobertura.

    Parâmetros
    ----------
    panorama : np.ndarray
        Panorama final após o blending.
    masks : list[np.ndarray]
        Máscaras das imagens já projetadas no canvas do panorama.
        Pixels válidos devem ser diferentes de zero.

    Retorna
    -------
    cropped : np.ndarray
        Panorama recortado.
    """
    if panorama is None or panorama.size == 0:
        raise ValueError("O panorama está vazio.")

    if not masks:
        raise ValueError("Nenhuma máscara foi fornecida.")

    height, width = panorama.shape[:2]
    coverage = np.zeros((height, width), dtype=bool)

    for mask in masks:
        if mask is None or mask.size == 0:
            continue

        if mask.shape[:2] != (height, width):
            raise ValueError(
                "Todas as máscaras devem ter as mesmas dimensões "
                "do panorama e estar alinhadas ao mesmo canvas."
            )

        coverage |= mask > 0

    if not coverage.any():
        raise ValueError("As máscaras não contêm pixels válidos.")

    # Maior retângulo totalmente coberto, usando histograma por linha.
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_rect = None

    for y in range(height):
        heights = np.where(coverage[y], heights + 1, 0)
        stack = []

        for x in range(width + 1):
            current = int(heights[x]) if x < width else 0
            start = x

            while stack and stack[-1][1] > current:
                left, bar_height = stack.pop()
                area = bar_height * (x - left)

                if area > best_area:
                    best_area = area
                    best_rect = (
                        left,
                        y - bar_height + 1,
                        x - left,
                        bar_height
                    )

                start = left

            if not stack or stack[-1][1] < current:
                stack.append((start, current))

    if best_rect is None:
        raise ValueError("Não foi possível encontrar um retângulo válido.")

    x, y, w, h = best_rect
    cropped = panorama[y:y+h, x:x+w].copy()

    return cropped