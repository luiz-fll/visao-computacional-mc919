import numpy as np

def estimate_affine(src_pts, dst_pts):
    """
    Estima uma transformação afim por mínimos quadrados.

    src_pts: (N, 2)
    dst_pts: (N, 2)

    Retorna:
        H_affine: (3, 3)
    """

    n = len(src_pts)

    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)

    """
    A:  [x1 y1 00 00 01 00]     b: [x1']
        [00 00 x1 y1 00 01]        [y1']
        ...
        [xn yn 00 00 01 00]        [xn']
        [00 00 xn yn 00 01]        [yn']
    
        [xi'] = [a b][xi] + [e]
        [yi']   [c d][yi]   [f]
    
        p = [a b c d e f]
    """
    for i, ((x, y), (xp, yp)) in enumerate(zip(src_pts, dst_pts)):
        
        A[2 * i] = [x, y, 0, 0, 1, 0]

        A[2 * i + 1] = [0, 0, x, y, 0, 1]

        b[2 * i] = xp
        b[2 * i + 1] = yp

    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    H = np.array([
        [p[0], p[1], p[4]],
        [p[2], p[3], p[5]],
        [0, 0,  1]
    ])

    return H

def project_points(H, src_pts):
    """
    Projeta pontos usando uma homografia.

    src_pts: (N, 2)
    H: (3, 3)

    Retorna:
        projected_pts: (N, 2)
    """

    n = len(src_pts)

    # Transformamos os pontos para coordenadas homogêneas:
    #
    # [x1 y1 1]
    # [x2 y2 1]
    # [...]
    # [xn yn 1]

    src_h = np.hstack([
        src_pts,
        np.ones((n, 1))
    ])

    projected_h = src_h @ H.T

    w = projected_h[:, 2]

    # projected_h:
    # [u1' v1' w1']
    # [u2' v2' w2']
    # [...]
    #
    # x' = u' / w'
    # y' = v' / w'

    projected_pts = np.full(
        (n, 2),
        np.nan,
        dtype=np.float64
    )

    valid = np.abs(w) > 1e-10

    projected_pts[valid] = (
        projected_h[valid, :2]
        / w[valid, np.newaxis]
    )

    return projected_pts

def compute_residuals(H, src_pts, dst_pts):
    """
    Calcula os resíduos da homografia.

    src_pts: (N, 2)
    dst_pts: (N, 2)
    H: (3, 3)

    Retorna:
        residuals: (N, 2)
    """
    projected_pts = project_points(H, src_pts)

    # residuals = [u_i - x_i']
    #             [v_i - y_i']
    residuals = projected_pts - dst_pts

    return residuals

def compute_jacobian(H, src_pts):
    """
    Calcula o Jacobiano da homografia.

    src_pts: (N, 2)
    H: (3, 3)

    Retorna:
        J: (2N, 8)

    Ordem dos parâmetros:
        [h11, h12, h13,
         h21, h22, h23,
         h31, h32]
    """
    n = len(src_pts)

    h11, h12, h13 = H[0]
    h21, h22, h23 = H[1]
    h31, h32 = H[2, 0], H[2, 1]

    J = np.zeros((2 * n, 8))

    for i, (x, y) in enumerate(src_pts):

        # Numerador da coordenada u:
        a = h11 * x + h12 * y + h13

        # Numerador da coordenada v:
        b = h21 * x + h22 * y + h23

        # Denominador:
        d = h31 * x + h32 * y + 1

        # Primeiro resíduo:
        #
        # r_u = u - x'
        #
        # Derivadas em relação aos 8 parâmetros:
        J[2 * i] = [x / d, y / d, 1 / d, 0, 0, 0, -x * a / d**2, -y * a / d**2]

        # Segundo resíduo:
        #
        # r_v = v - y'
        #
        # Derivadas em relação aos 8 parâmetros:
        J[2 * i + 1] = [0, 0, 0, x / d, y / d, 1 / d, -x * b / d**2, -y * b / d**2]

    return J

def gauss_newton(H_initial, src_pts, dst_pts,
                 max_iterations=100,
                 tolerance=1e-6):
    """
    Refina uma homografia usando Gauss-Newton.

    H_initial: (3, 3)
        Estimativa inicial da homografia.

    src_pts: (N, 2)
    dst_pts: (N, 2)

    Retorna:
        H: (3, 3)
    """
    H = H_initial.copy()

    for iteration in range(max_iterations):

        # residuals = [u1 - x1']
        #             [v1 - y1']
        #             [u2 - x2']
        #             [v2 - y2']
        #             [...]

        residuals = compute_residuals(H, src_pts, dst_pts)

        # J Δ = -r
        # r = [r_u1, r_v1, r_u2, r_v2, ...]
        r = residuals.ravel()
        J = compute_jacobian(H, src_pts)
        delta, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)

        # delta = [Δh11 Δh12 Δh13 Δh21 Δh22 Δh23 Δh31 Δh32]

        H[:2, :] += delta[:6].reshape(2, 3)
        H[2, 0] += delta[6]
        H[2, 1] += delta[7]

        if np.linalg.norm(delta) < tolerance:
            break

    return H

def estimate_homography(src_pts, dst_pts):
    """
    Estima uma homografia para um único par de imagens.

    Etapas:
        1. Estima uma transformação afim por mínimos quadrados.
        2. Usa a afim como aproximação inicial.
        3. Refina a homografia usando Gauss-Newton.

    src_pts: (N, 2)
    dst_pts: (N, 2)

    Retorna:
        H: (3, 3)
    """
    if len(src_pts) < 4:
        raise ValueError("São necessários pelo menos 4 pares de pontos.")
    
    H_affine = estimate_affine(src_pts, dst_pts)
    H = gauss_newton(H_affine, src_pts, dst_pts)

    return H

def compute_reprojection_errors(H, src_pts, dst_pts):
    """
    Calcula o erro de reprojeção para cada correspondência.

    src_pts: pontos na imagem de origem, shape (N, 2)
    dst_pts: pontos na imagem de destino, shape (N, 2)

    Retorna:
        errors: erro de reprojeção de cada ponto, shape (N,)
    """

    residuals = compute_residuals(H, src_pts, dst_pts)
    errors = np.linalg.norm(residuals, axis=1)

    return errors

def compute_reprojection_errors(H, src_pts, dst_pts):
    """
    Calcula o erro de reprojeção de cada correspondência.
    """

    projected_pts = project_points(H, src_pts)
    residuals = projected_pts - dst_pts

    errors = np.linalg.norm(residuals, axis=1)

    # Projeções inválidas são consideradas
    # automaticamente como muito ruins.
    errors[~np.isfinite(errors)] = np.inf

    return errors

def ransac_homography(src_pts, dst_pts, num_iterations=1000, threshold=3.0):
    """
    Estima uma homografia robusta usando RANSAC.

    src_pts: pontos da imagem de origem, shape (N, 2)
    dst_pts: pontos da imagem de destino, shape (N, 2)

    Retorna:
        H_final: homografia estimada
        best_inliers: máscara booleana dos inliers
        best_errors: erros de reprojeção da melhor hipótese
    """
    n_points = len(src_pts)

    if n_points < 4:
        raise ValueError("São necessários pelo menos 4 pontos.")

    best_H = None
    best_inliers = None
    best_errors = None
    best_num_inliers = 0

    for iteration in range(num_iterations):
        sample_indices = np.random.choice(n_points, size=4, replace=False)
        sample_src = src_pts[sample_indices]
        sample_dst = dst_pts[sample_indices]

        try:
            H = estimate_homography(sample_src,sample_dst)
        except np.linalg.LinAlgError:
            # Caso a configuração dos pontos seja degenerada
            continue

        errors = compute_reprojection_errors(H, src_pts, dst_pts)
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_H = H
            best_inliers = inliers
            best_errors = errors

    if best_H is None:
        raise RuntimeError("O RANSAC não conseguiu encontrar uma homografia.")

    H_final = estimate_homography(src_pts[best_inliers], dst_pts[best_inliers])

    final_errors = compute_reprojection_errors(H_final, src_pts, dst_pts)
    final_inliers = final_errors < threshold

    return H_final, final_inliers, final_errors

def estimate_all_homographies(lowe_matrix, keypoints, order, num_iterations=1000, threshold=3.0):
    """
    Estima as homografias entre todos os pares vizinhos
    na ordem determinada pelo processo de ordenação.

    lowe_matrix:
        Matriz NxN contendo os matches após o teste de Lowe.

        Cada célula lowe_matrix[i][j] contém uma lista
        de objetos cv2.DMatch.

    keypoints:
        Lista de keypoints das imagens.

        keypoints[i] contém os cv2.KeyPoint da imagem i.

    order:
        Ordem das imagens obtida pelo processo de ordenação.

        Exemplo:
            [2, 0, 3, 1]

        significa que a sequência é:
            imagem 2 -> imagem 0 -> imagem 3 -> imagem 1

    Retorna:
        homographies:
            Lista das homografias entre pares vizinhos.

        metrics:
            Lista com as métricas de cada par.
    """
    homographies = []
    metrics = []

    for position in range(len(order) - 1):

        i = order[position]
        j = order[position + 1]

        print(f"Processando: imagem {i} -> imagem {j}")

        matches = lowe_matrix[i][j]
        if len(matches) < 4:
            raise ValueError(f"Poucos matches entre as imagens {i} e {j}: {len(matches)}")

        # Transformar os matches em coordenadas
        src_pts = np.float32([
            keypoints[i][match.queryIdx].pt
            for match in matches
        ])

        dst_pts = np.float32([
            keypoints[j][match.trainIdx].pt
            for match in matches
        ])

        H, inliers, errors = ransac_homography(src_pts, dst_pts, num_iterations=num_iterations, threshold=threshold)

        num_matches = len(matches)
        num_inliers = np.sum(inliers)
        inlier_rate = num_inliers / num_matches
        mean_reprojection_error = errors[inliers].mean()

        homographies.append(H)

        metrics.append({
            "src_index": i,
            "dst_index": j,
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_rate": inlier_rate,
            "mean_reprojection_error": mean_reprojection_error
        })

    return homographies, metrics