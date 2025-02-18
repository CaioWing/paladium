import cv2


def preprocess_license_plate(image):
    """Melhora o contraste para OCR."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    return image
