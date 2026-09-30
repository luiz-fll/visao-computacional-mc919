import cv2
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

def plot_image_grid(images, title="Imagens", layout=(2, 3), size=(10, 6)):
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

def plot_matrix(matrix):
    plt.figure(figsize=(8, 6))

    plt.imshow(
        matrix,
        cmap="viridis"
    )

    plt.colorbar(label="Número de matches")

    plt.xlabel("Imagem")
    plt.ylabel("Imagem")
    plt.title("Matriz de conectividade")
    plt.show()
    plt.close()