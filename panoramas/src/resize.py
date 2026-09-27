import argparse
import cv2
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def resize_images(input_dir: Path, output_dir: Path, scale: float) -> None:
    if scale <= 0:
        raise ValueError("O fator de escala deve ser maior que 0.")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print(f"Nenhuma imagem encontrada em: {input_dir}")
        return

    for image_path in sorted(images):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Não foi possível ler: {image_path}")
            continue

        height, width = image.shape[:2]

        new_width = round(width * scale)
        new_height = round(height * scale)

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), resized)

        print(f"{image_path.name}: {width}x{height} -> {new_width}x{new_height}")


def main():
    parser = argparse.ArgumentParser(
        description="Redimensiona imagens de um diretório."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Diretório contendo as imagens originais.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Diretório onde as imagens redimensionadas serão salvas.",
    )

    parser.add_argument(
        "--scale",
        required=True,
        type=float,
        help="Fator de escala. Ex.: 0.5 reduz pela metade.",
    )

    args = parser.parse_args()

    resize_images(
        args.input,
        args.output,
        args.scale,
    )


if __name__ == "__main__":
    main()