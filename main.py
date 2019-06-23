import cv2
from matplotlib import pyplot as plt


def convertBGRToRGB(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def detect_faces(face_cascade_detector, bgr_scale_img, scaling_factor=1.1):
    bgr_scale_img_copy = bgr_scale_img.copy()
    gray_scale_img = cv2.cvtColor(bgr_scale_img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_detector.detectMultiScale(gray_scale_img, scaleFactor=scaling_factor, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.rectangle(bgr_scale_img_copy, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    return bgr_scale_img_copy


test_img = cv2.imread('data/test1.jpg')
plt.imshow(convertBGRToRGB(test_img))
plt.show()

haar_cascade_face_detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
faces_detected = detect_faces(haar_cascade_face_detector, test_img)
plt.imshow(convertBGRToRGB(faces_detected))
plt.show()
