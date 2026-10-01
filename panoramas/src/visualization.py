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
        final_matrix = matrix[np.ix_(order, order)]
    else:
        final_matrix = matrix

    plt.imshow(
        final_matrix,
        cmap="viridis"
    )

    plt.colorbar(label="Número de matches")

    plt.xlabel("Imagem")
    plt.ylabel("Imagem")
    plt.title(title)
    plt.show()
    plt.close()