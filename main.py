import cv2


def convertToRGB(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
