import cv2

def detect_sift(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def detect_orb(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors

def draw_keypoints(images, method="sift"):
    images_with_keypoints = []
    if method.lower() == "sift":
        detector = detect_sift
    elif method.lower() == "orb":
        detector = detect_orb

    print(f"Detector: {method}")
    for image in images:
        keypoints, descriptors = detector(image)
        print("Número de keypoints:", len(keypoints))
        print("Formato dos descritores:", descriptors.shape)

        image_keypoints = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        images_with_keypoints.append({
            "image": image_keypoints,
            "keypoints": keypoints,
            "descriptors": descriptors
        })

    return images_with_keypoints