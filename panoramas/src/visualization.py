import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image, title="", max_width=1600):
    h, w = image.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_width = int(w * scale)
        new_height = int(h * scale)

        final_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        final_image = image
    
    plt.figure(figsize=(16, 8))

    plt.imshow(
        cv2.cvtColor(
            final_image,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")
    plt.title(title)
    plt.show()
    plt.close()

def plot_image_grid(images, title="Imagens", layout=(2, 4), size=(12, 6)):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=size)
    fig.suptitle(title)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Imagem {i}')

def plot_matches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches,
    title=""
):
    result = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(18, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
    plt.close()

def plot_matrix(matrix, title="Matriz", order=None):
    plt.figure(figsize=(8, 6))

    if order:
        plt.xticks(
            range(len(order)),
            order
        )

        plt.yticks(
            range(len(order)),
            order
        )

    plt.imshow(
        matrix,
        cmap="viridis"
    )

    plt.colorbar(label="Número de matches")

    plt.xlabel("Imagem")
    plt.ylabel("Imagem")
    plt.title(title)
    plt.show()
    plt.close()

def plot_iterations(image_src, image_dst, results):
    src_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    dst_rgb = cv2.cvtColor(image_dst, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))

    for row, result in enumerate(results):
        H = result["H"]
        height, width = image_dst.shape[:2]

        # Transformar a imagem origem para o sistema da imagem destino
        warped = cv2.warpPerspective(image_src, H, (width, height))
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(dst_rgb)
        axes[row, 0].set_title(f"Destino\n {result['iterations']} iterações")
        axes[row, 0].axis("off")

        # Imagem origem deformada
        axes[row, 1].imshow(warped_rgb)
        axes[row, 1].set_title(f"Warp da origem\n inliers = {result["metrics"]['num_inliers']}")
        axes[row, 1].axis("off")

        # Sobreposição
        axes[row, 2].imshow(dst_rgb)
        axes[row, 2].imshow(warped_rgb, alpha=0.5)

        axes[row, 2].set_title(
            f"Sobreposição\n"
            f"inlier rate = {result["metrics"]['inlier_rate']:.2%}\n"
            f"erro médio = {result["metrics"]['mean_reprojection_error']:.2f} px"
        )

        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()