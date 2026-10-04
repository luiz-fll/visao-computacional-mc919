"""
Pipeline completo de criação de panoramas.

Uso típico:

    from panorama import create_panorama
    import cv2

    images = [cv2.imread(f"img{i}.jpg") for i in range(6)]
    panorama, info = create_panorama(images, focal_length=800)
    cv2.imwrite("panorama.jpg", panorama)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.features import detect_sift, detect_orb
from src.matching import match_all_images
from src.ordering import infer_order
from src.homography import estimate_all_homographies
from src.composition import cylindrical_stitching
from src.blending import optimal_seam_blending, feather_blending


def create_panorama(
    images: List[np.ndarray],
    *,
    # ---- Detecção de features ----
    feature_method: str = "sift",
    # ---- Matching ----
    lowe_ratio: float = 0.75,
    # ---- Ordenação ----
    outlier_threshold: float = 0.5,
    # ---- Homografia / RANSAC ----
    ransac_iterations: int = 1000,
    ransac_threshold: float = 3.0,
    # ---- Composição cilíndrica ----
    focal_length: float = 800.0,
    # ---- Blending ----
    blending_method: str = "optimal_seam",
    feather_width: int = 30,
    # ---- Controle ----
    return_intermediate: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cria um panorama a partir de uma lista de imagens (do início ao fim).

    Parameters
    ----------
    images : list[np.ndarray]
        Lista de imagens BGR (já carregadas com cv2.imread).
        Não precisam estar ordenadas.

    feature_method : str
        Detector de features: "sift" (mais preciso) ou "orb" (mais rápido).

    lowe_ratio : float
        Limiar do teste de razão de Lowe (tipicamente 0.7 ~ 0.8).

    outlier_threshold : float
        Limiar do score bilateral médio usado na detecção de outliers.
        Valores menores são mais permissivos.

    ransac_iterations : int
        Número de iterações do RANSAC na estimativa de homografias.

    ransac_threshold : float
        Limiar de erro de reprojeção (em pixels) para considerar um ponto inlier.

    focal_length : float
        Distância focal em pixels usada na projeção cilíndrica.
        Valor típico: 0.7 ~ 1.2 × largura da imagem.

    blending_method : str
        Método de blending:
            - "optimal_seam" : costura ótima + feathering local (recomendado)
            - "feather"      : feather blending multi-imagem clássico

    feather_width : int
        Largura da transição suave no blending (usado em ambos os métodos).

    return_intermediate : bool
        Se True, o dicionário de informações contém também as etapas intermediárias
        (keypoints, matches, ordem, homografias, imagens warped, máscaras…).

    verbose : bool
        Se True, imprime progresso no terminal.

    Returns
    -------
    panorama : np.ndarray
        Imagem final do panorama (BGR, uint8).

    info : dict
        Metadados e, opcionalmente, resultados intermediários.
        Sempre contém pelo menos:
            - order
            - connectivity_matrix
            - homography_metrics
            - composition_info
    """
    if len(images) < 2:
        raise ValueError("É necessário fornecer pelo menos duas imagens.")

    n = len(images)

    # ------------------------------------------------------------------
    # 1. Detecção de features
    # ------------------------------------------------------------------
    if verbose:
        print(f"[Features] Detectando keypoints ({feature_method.upper()})")

    if feature_method.lower() == "sift":
        detector = detect_sift
    elif feature_method.lower() == "orb":
        detector = detect_orb
    else:
        raise ValueError("feature_method deve ser 'sift' ou 'orb'")

    keypoints_list = []
    descriptors_list = []

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        kpts, desc = detector(gray)
        keypoints_list.append(kpts)
        descriptors_list.append(desc)
        if verbose:
            n_kpts = len(kpts) if kpts is not None else 0
            print(f"[Features]   Imagem {i}: {n_kpts} keypoints")

    # ------------------------------------------------------------------
    # 2. Matching bilateral + teste de Lowe + cross-check
    # ------------------------------------------------------------------
    if verbose:
        print(f"[Matching] Matching bilateral (ratio={lowe_ratio})")

    matches_matrix, lowe_matrix = match_all_images(
        descriptors_list,
        method=feature_method,
        ratio=lowe_ratio,
    )

    if verbose:
        total_matches = sum(len(lowe_matrix[i][j]) for i in range(n) for j in range(i + 1, n))
        print(f"[Matching]   Total de matches mútuos: {total_matches}")

    # ------------------------------------------------------------------
    # 3. Detecção de outliers e ordenação
    # ------------------------------------------------------------------
    if verbose:
        print(f"[Ordering] Ordenando imagens (outlier_threshold={outlier_threshold})")

    order, connectivity_matrix, filtered_matrix = infer_order(
        lowe_matrix,
        outlier_threshold=outlier_threshold,
    )

    if verbose:
        print(f"[Ordering]   Ordem encontrada: {order}")
        if len(order) < n:
            outliers = sorted(set(range(n)) - set(order))
            print(f"[Ordering]   Outliers removidos: {outliers}")

    if len(order) < 2:
        raise RuntimeError(
            "Menos de duas imagens restaram após a detecção de outliers. "
            "Tente diminuir o outlier_threshold."
        )

    # ------------------------------------------------------------------
    # 4. Estimativa de homografias entre vizinhos
    # ------------------------------------------------------------------
    if verbose:
        print(f"[Homography] Estimando homografias (RANSAC {ransac_iterations} it., threshold={ransac_threshold} px)")

    homographies, homography_metrics = estimate_all_homographies(
        lowe_matrix,
        keypoints_list,
        order,
        num_iterations=ransac_iterations,
        threshold=ransac_threshold,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 5. Projeção cilíndrica
    # ------------------------------------------------------------------
    if verbose:
        print(f"[Composition] Projeção cilíndrica (f = {focal_length:.1f} px)")

    warped_images, masks, composition_info = cylindrical_stitching(
        images,
        order,
        homographies,
        focal_length=focal_length,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 6. Blending
    # ------------------------------------------------------------------
    if blending_method == "optimal_seam":
        panorama = optimal_seam_blending(
            warped_images,
            masks,
            feather_width=feather_width,
            verbose=verbose,
        )
    elif blending_method == "feather":
        panorama = feather_blending(
            warped_images,
            masks,
            verbose=verbose,
        )
    else:
        raise ValueError("blending_method deve ser 'optimal_seam' ou 'feather'")

    if verbose:
        h, w = panorama.shape[:2]
        print(f"[Panorama] Finalizado: {w} × {h}")

    # ------------------------------------------------------------------
    # Informações de retorno
    # ------------------------------------------------------------------
    info: Dict[str, Any] = {
        "order": order,
        "connectivity_matrix": connectivity_matrix,
        "filtered_connectivity": filtered_matrix,
        "homography_metrics": homography_metrics,
        "composition_info": composition_info,
        "feature_method": feature_method,
        "focal_length": focal_length,
        "blending_method": blending_method,
    }

    if return_intermediate:
        info.update({
            "keypoints": keypoints_list,
            "descriptors": descriptors_list,
            "matches_matrix": matches_matrix,
            "lowe_matrix": lowe_matrix,
            "homographies": homographies,
            "warped_images": warped_images,
            "masks": masks,
        })

    return panorama, info


# ----------------------------------------------------------------------
# Exemplo de uso (executável diretamente)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Cria um panorama a partir de um diretório de imagens.")
    parser.add_argument("input_dir", type=str, help="Diretório contendo as imagens")
    parser.add_argument("-o", "--output", type=str, default="panorama.jpg", help="Arquivo de saída")
    parser.add_argument("--focal", type=float, default=800.0, help="Distância focal em pixels")
    parser.add_argument("--method", choices=["sift", "orb"], default="sift", help="Detector de features")
    parser.add_argument("--blending", choices=["optimal_seam", "feather"], default="optimal_seam")
    parser.add_argument("--feather-width", type=int, default=30)
    parser.add_argument("--lowe-ratio", type=float, default=0.75)
    parser.add_argument("--outlier-threshold", type=float, default=0.5)
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true", help="Não imprime progresso")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if len(paths) < 2:
        raise SystemExit(f"Poucas imagens encontradas em {input_dir}")

    print(f"[Panorama] Carregando {len(paths)} imagens de {input_dir}")
    images = [cv2.imread(str(p)) for p in paths]
    for p, img in zip(paths, images):
        if img is None:
            raise SystemExit(f"[Panorama] Falha ao ler {p}")

    panorama, info = create_panorama(
        images,
        feature_method=args.method,
        lowe_ratio=args.lowe_ratio,
        outlier_threshold=args.outlier_threshold,
        ransac_iterations=args.ransac_iterations,
        ransac_threshold=args.ransac_threshold,
        focal_length=args.focal,
        blending_method=args.blending,
        feather_width=args.feather_width,
        verbose=not args.quiet,
    )

    cv2.imwrite(args.output, panorama)
    print(f"[Panorama] Salvo em: {args.output}")
    print(f"[Panorama] Ordem utilizada: {info['order']}")
