"""
Funções de visualização para inspeção do pipeline de panoramas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image, title="", max_width=1600):
    """
    Exibe uma única imagem BGR, redimensionando se necessário para caber na tela.
    """
    h, w = image.shape[:2]

    if w > max_width:
        scale = max_width / w
        final_image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        final_image = image

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
    plt.close()


def plot_image_grid(images, title="Imagens", layout=(2, 4), size=(12, 6)):
    """
    Exibe várias imagens em uma grade.
    """
    fig, axes = plt.subplots(layout[0], layout[1], figsize=size)
    fig.suptitle(title)

    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Imagem {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_matches(image1, keypoints1, image2, keypoints2, matches, title=""):
    """
    Desenha matches entre dois pares de imagens.
    """
    result = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(18, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
    plt.close()


def plot_matrix(matrix, title="Matriz", order=None):
    """
    Visualiza uma matriz de conectividade (número de matches).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="Número de matches")

    if order is not None:
        plt.xticks(range(len(order)), order)
        plt.yticks(range(len(order)), order)

    plt.xlabel("Imagem")
    plt.ylabel("Imagem")
    plt.title(title)
    plt.show()
    plt.close()


def plot_iterations(image_src, image_dst, results):
    """
    Visualiza o resultado de várias iterações de estimativa de homografia.

    Parameters
    ----------
    image_src, image_dst : np.ndarray
    results : list[dict]
        Cada dicionário deve conter as chaves:
            - "H"
            - "iterations"
            - "metrics" (com "num_inliers", "inlier_rate", "mean_reprojection_error")
    """
    src_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    dst_rgb = cv2.cvtColor(image_dst, cv2.COLOR_BGR2RGB)

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, result in enumerate(results):
        H = result["H"]
        height, width = image_dst.shape[:2]
        warped = cv2.warpPerspective(image_src, H, (width, height))
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(dst_rgb)
        axes[row, 0].set_title(f"Destino\n{result['iterations']} iterações")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(warped_rgb)
        axes[row, 1].set_title(f"Warp da origem\ninliers = {result['metrics']['num_inliers']}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(dst_rgb)
        axes[row, 2].imshow(warped_rgb, alpha=0.5)
        axes[row, 2].set_title(
            f"Sobreposição\n"
            f"inlier rate = {result['metrics']['inlier_rate']:.2%}\n"
            f"erro médio = {result['metrics']['mean_reprojection_error']:.2f} px"
        )
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
