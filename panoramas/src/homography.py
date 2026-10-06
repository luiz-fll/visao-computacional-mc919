"""
Estimativa de homografias.

Fluxo principal:
1. Estimativa inicial por transformação afim (mínimos quadrados).
2. Refinamento não-linear com Levenberg-Marquardt (scipy.optimize.least_squares).
3. RANSAC para robustez a outliers.
4. Estimativa de todas as homografias consecutivas na ordem determinada.
"""

import numpy as np
from scipy.optimize import least_squares


def estimate_affine(src_pts, dst_pts):
    """
    Estima uma transformação afim por mínimos quadrados.

    Modelo:
        [x']   [a  b  e] [x]
        [y'] = [c  d  f] [y]
        [1 ]   [0  0  1] [1]

    Parameters
    ----------
    src_pts : np.ndarray, shape (N, 2)
    dst_pts : np.ndarray, shape (N, 2)

    Returns
    -------
    H_affine : np.ndarray, shape (3, 3)
    """
    n = len(src_pts)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)

    for i, ((x, y), (xp, yp)) in enumerate(zip(src_pts, dst_pts)):
        A[2 * i]     = [x, y, 0, 0, 1, 0]
        A[2 * i + 1] = [0, 0, x, y, 0, 1]
        b[2 * i]     = xp
        b[2 * i + 1] = yp

    p, *_ = np.linalg.lstsq(A, b, rcond=None)

    H = np.array([
        [p[0], p[1], p[4]],
        [p[2], p[3], p[5]],
        [0.0,  0.0,  1.0]
    ])
    return H


def project_points(H, src_pts):
    """
    Projeta pontos através de uma homografia.

    Parameters
    ----------
    H : np.ndarray, shape (3, 3)
    src_pts : np.ndarray, shape (N, 2)

    Returns
    -------
    projected_pts : np.ndarray, shape (N, 2)
        Pontos inválidos (w ≈ 0) recebem NaN.
    """
    n = len(src_pts)
    src_h = np.hstack([src_pts, np.ones((n, 1))])
    projected_h = src_h @ H.T

    w = projected_h[:, 2]
    projected_pts = np.full((n, 2), np.nan, dtype=np.float64)

    valid = np.abs(w) > 1e-10
    projected_pts[valid] = projected_h[valid, :2] / w[valid, np.newaxis]
    return projected_pts


def _homography_residuals(params, src_pts, dst_pts):
    """
    Função de resíduos usada pelo least_squares.

    params = [h11, h12, h13, h21, h22, h23, h31, h32]
    (h33 é fixado em 1)
    """
    H = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0]
    ])
    projected = project_points(H, src_pts)
    residuals = (projected - dst_pts).ravel()
    # Substitui NaNs por um valor grande para não quebrar o otimizador
    residuals[~np.isfinite(residuals)] = 1e6
    return residuals


def refine_homography(H_initial, src_pts, dst_pts):
    """
    Refina uma homografia usando Levenberg-Marquardt (scipy).

    Parameters
    ----------
    H_initial : np.ndarray, shape (3, 3)
        Estimativa inicial (normalmente a transformação afim).
    src_pts : np.ndarray, shape (N, 2)
    dst_pts : np.ndarray, shape (N, 2)

    Returns
    -------
    H : np.ndarray, shape (3, 3)
        Homografia refinada.
    """
    # Parametrização com 8 graus de liberdade (h33 = 1)
    x0 = np.array([
        H_initial[0, 0], H_initial[0, 1], H_initial[0, 2],
        H_initial[1, 0], H_initial[1, 1], H_initial[1, 2],
        H_initial[2, 0], H_initial[2, 1]
    ])

    result = least_squares(
        _homography_residuals,
        x0,
        args=(src_pts, dst_pts),
        method="lm",          # Levenberg-Marquardt
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=200,
    )

    p = result.x
    H = np.array([
        [p[0], p[1], p[2]],
        [p[3], p[4], p[5]],
        [p[6], p[7], 1.0]
    ])
    return H


def estimate_homography(src_pts, dst_pts):
    """
    Estima uma homografia para um conjunto de correspondências.

    Etapas:
    1. Estimativa afim por mínimos quadrados (inicialização).
    2. Refinamento não-linear com Levenberg-Marquardt.

    Parameters
    ----------
    src_pts : np.ndarray, shape (N, 2)
    dst_pts : np.ndarray, shape (N, 2)

    Returns
    -------
    H : np.ndarray, shape (3, 3)
    """
    if len(src_pts) < 4:
        raise ValueError("São necessários pelo menos 4 pares de pontos.")

    H_affine = estimate_affine(src_pts, dst_pts)
    H = refine_homography(H_affine, src_pts, dst_pts)
    return H


def compute_reprojection_errors(H, src_pts, dst_pts):
    """
    Calcula o erro de reprojeção (distância euclidiana) de cada correspondência.

    Parameters
    ----------
    H : np.ndarray, shape (3, 3)
    src_pts : np.ndarray, shape (N, 2)
    dst_pts : np.ndarray, shape (N, 2)

    Returns
    -------
    errors : np.ndarray, shape (N,)
        Erros de reprojeção. Projeções inválidas recebem np.inf.
    """
    projected = project_points(H, src_pts)
    residuals = projected - dst_pts
    errors = np.linalg.norm(residuals, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def ransac_homography(src_pts, dst_pts, num_iterations=100, threshold=3.0):
    """
    Estima uma homografia robusta usando RANSAC.

    Parameters
    ----------
    src_pts : np.ndarray, shape (N, 2)
    dst_pts : np.ndarray, shape (N, 2)
    num_iterations : int
        Número de iterações do RANSAC.
    threshold : float
        Limiar de erro de reprojeção (em pixels) para considerar um ponto inlier.

    Returns
    -------
    H_final : np.ndarray, shape (3, 3)
        Homografia estimada apenas com os inliers da melhor hipótese.
    final_inliers : np.ndarray, shape (N,), dtype=bool
        Máscara dos inliers da estimativa final.
    final_errors : np.ndarray, shape (N,)
        Erros de reprojeção da estimativa final.
    """
    n_points = len(src_pts)
    if n_points < 4:
        raise ValueError("São necessários pelo menos 4 pontos.")

    best_H = None
    best_inliers = None
    best_num_inliers = 0

    for _ in range(num_iterations):
        sample_idx = np.random.choice(n_points, size=4, replace=False)
        try:
            H = estimate_homography(src_pts[sample_idx], dst_pts[sample_idx])
        except (np.linalg.LinAlgError, ValueError):
            continue

        errors = compute_reprojection_errors(H, src_pts, dst_pts)
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_H = H
            best_inliers = inliers

    if best_H is None:
        raise RuntimeError("RANSAC não conseguiu encontrar uma homografia válida.")

    # Reestimativa final usando todos os inliers da melhor hipótese
    H_final = estimate_homography(src_pts[best_inliers], dst_pts[best_inliers])
    final_errors = compute_reprojection_errors(H_final, src_pts, dst_pts)
    final_inliers = final_errors < threshold

    return H_final, final_inliers, final_errors


def estimate_all_homographies(lowe_matrix, keypoints, order,
                              num_iterations=1000, threshold=3.0,
                              verbose=True):
    """
    Estima as homografias entre todos os pares vizinhos na ordem dada.

    Parameters
    ----------
    lowe_matrix : np.ndarray (dtype=object)
        Matriz N×N com matches após teste de Lowe + cross-check.
    keypoints : list[list[cv2.KeyPoint]]
        keypoints[i] contém os keypoints da imagem i.
    order : list[int]
        Ordem das imagens (ex.: [2, 0, 3, 1]).
    num_iterations : int
        Iterações do RANSAC.
    threshold : float
        Limiar de inlier do RANSAC (pixels).
    verbose : bool
        Se True, imprime o progresso de cada par.

    Returns
    -------
    homographies : list[np.ndarray]
        Lista de homografias 3×3. homographies[k] leva order[k] → order[k+1].
    metrics : list[dict]
        Métricas de qualidade de cada par.
    """
    homographies = []
    metrics = []

    for pos in range(len(order) - 1):
        i = order[pos]
        j = order[pos + 1]

        if verbose:
            print(f"[Homography] Estimando {i} → {j}")

        matches = lowe_matrix[i][j]
        if len(matches) < 4:
            raise ValueError(f"Poucos matches entre as imagens {i} e {j}: {len(matches)}")

        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints[j][m.trainIdx].pt for m in matches])

        H, inliers, errors = ransac_homography(
            src_pts, dst_pts,
            num_iterations=num_iterations,
            threshold=threshold,
        )

        num_matches = len(matches)
        num_inliers = int(np.sum(inliers))
        inlier_rate = num_inliers / num_matches
        mean_error = float(errors[inliers].mean()) if num_inliers > 0 else np.inf

        if verbose:
            print(
                f"[Homography]   {i} → {j}: "
                f"{num_inliers}/{num_matches} inliers ({inlier_rate:.1%}), "
                f"erro médio = {mean_error:.2f} px"
            )

        homographies.append(H)
        metrics.append({
            "src_index": i,
            "dst_index": j,
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_rate": inlier_rate,
            "mean_reprojection_error": mean_error,
        })

    return homographies, metrics
