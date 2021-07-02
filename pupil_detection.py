import numpy as np
import cv2
from matplotlib import pyplot as plt

#얼굴분류기, 눈 분류기 불러오기
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# 이미지 파일 불러오기
img = cv2.imread('test.jpeg')

# 이미지 출력하기  
plt.figure(figsize=(7,10))
plt.imshow(img[:,:,::-1]) 
plt.show()
