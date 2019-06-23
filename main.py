import cv2
from matplotlib import pyplot as plt
from time import time


def convertBGRToRGB(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def detect_faces(face_cascade_detector, bgr_scale_img, scaling_factor=1.1):
    bgr_scale_img_copy = bgr_scale_img.copy()
    gray_scale_img = cv2.cvtColor(bgr_scale_img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_detector.detectMultiScale(gray_scale_img, scaleFactor=scaling_factor, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.rectangle(bgr_scale_img_copy, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    return bgr_scale_img_copy


test_img = cv2.imread('data/test5.jpg')
plt.imshow(convertBGRToRGB(test_img))
plt.show()

haar_cascade_face_detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
t1 = time()
haar_detected_faces = detect_faces(haar_cascade_face_detector, test_img)
t2 = time()
haar_cascade_detection_time = t2 - t1

lbp_cascade_face_detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/lbpcascade_frontalface_improved.xml')
t1 = time()
lbp_detected_faces = detect_faces(haar_cascade_face_detector, test_img)
t2 = time()
lbp_cascade_detection_time = t2 - t1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title('Haar Cascade Detection Time: ' + str(round(haar_cascade_detection_time, 3)) + ' secs')
ax1.imshow(convertBGRToRGB(haar_detected_faces))
ax2.set_title('LBP Cascade Detection Time: ' + str(round(lbp_cascade_detection_time, 3)) + ' secs')
ax2.imshow(convertBGRToRGB(lbp_detected_faces))
plt.show()
