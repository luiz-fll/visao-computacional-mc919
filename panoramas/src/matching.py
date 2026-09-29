import cv2

def match_descriptors(descriptors1, descriptors2, method="sift"):
    if method.lower() == "sift":
        norm = cv2.NORM_L2
    elif method.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("Método deve ser 'sift' ou 'orb'.")

    matcher = cv2.BFMatcher(norm)

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    return matches


def lowe_ratio_test(matches, ratio=0.75):
    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        best, second = pair

        if best.distance < ratio * second.distance:
            good_matches.append(best)

    return good_matches


def match_all_images(descriptors, method="sift", ratio=0.75):
    matches_dict = {}
    lowe_matches_dict = {}

    n = len(descriptors)

    for i in range(n):
        for j in range(i + 1, n):

            matches = match_descriptors(
                descriptors[i],
                descriptors[j],
                method=method
            )

            lowe_matches = lowe_ratio_test(
                matches,
                ratio=ratio
            )

            matches_dict[(i, j)] = [pair[0] for pair in matches]
            lowe_matches_dict[(i, j)] = lowe_matches

    return matches_dict, lowe_matches_dict