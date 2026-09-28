import cv2
import numpy as np


def estimate_homography(
    keypoints1,
    keypoints2,
    good_matches,
    reproj_threshold=5.0
):
    """
    Estima a homografia entre duas imagens usando RANSAC.

    Retorna:
        H: matriz de homografia 3x3
        mask: máscara indicando quais matches são inliers
    """

    if len(good_matches) < 4:
        return None, None

    points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        reproj_threshold
    )

    return H, mask

def inlier_rate(mask):
    if mask is None:
        return 0.0

    return float(mask.ravel().mean())

def reprojection_error(keypoints1, keypoints2, good_matches, H, mask):
    if H is None or mask is None:
        return None

    points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    projected_points = cv2.perspectiveTransform(points1, H)

    errors = np.linalg.norm(
        projected_points - points2,
        axis=2
    ).ravel()

    inliers = mask.ravel().astype(bool)

    if not np.any(inliers):
        return None

    return float(errors[inliers].mean())