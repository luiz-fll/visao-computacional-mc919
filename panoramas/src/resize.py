"""
Utilitário simples para redimensionar um diretório de imagens.
"""
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def resize_images(input_dir: Path, output_dir: Path, scale: float) -> None:
    """
    Redimensiona todas as imagens de um diretório por um fator de escala.

    Parameters
    ----------
    input_dir : Path
        Diretório de entrada.
    output_dir : Path
        Diretório de saída (criado se não existir).
    scale : float
        Fator de escala (> 0). Valores < 1 reduzem a imagem.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"O diretório de entrada '{input_dir}' não existe.")
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
        new_width  = round(width * scale)
        new_height = round(height * scale)

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), resized)
        print(f"{image_path.name}: {width}×{height} → {new_width}×{new_height}")