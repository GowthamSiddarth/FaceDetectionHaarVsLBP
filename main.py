import cv2
from matplotlib import pyplot as plt


def convertToRGB(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


test1 = cv2.imread('data/test1.jpg')
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.show()

haar_cascade_face_detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
faces = haar_cascade_face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print(len(faces))

