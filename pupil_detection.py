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

# 이미지에서 얼굴 찾기
faces = face_cascade.detectMultiScale(img)

temp = img.copy()
for (x,y,w,h) in faces:
 # 얼굴 바운딩박스 그리기
 cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
 roi_color = temp[y:y+h, x:x+w]
 # 얼굴에서 눈 찾기
 eyes = eye_cascade.detectMultiScale(roi_color)
 for (ex,ey,ew,eh) in eyes:
 # 눈 바운딩 박스 그리기
 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#이미지 출력하기
plt.figure(figsize=(7,10))
plt.imshow(temp[:,:,::-1])
plt.show()

#scaleFactor

'''
이미지를 윈도우와 매칭 시킬 때 이미지 사이즈를 조절하는 변수
예를 들어 scaleFactor = 1.2인 경우 이미지를 1/1.2로 축소시켜 이미지를 윈도우와 매칭시킨다.(윈도우 사이즈는 고정되어있음)
많은 경우 값이 작을 수록 정확도가 높다
'''

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
#faces = face_cascade.detectMultiScale(img, scaleFactor = 1.1)
#faces2 = face_cascade.detectMultiScale(img, scaleFactor = 1.8)
#temp = img.copy()
#temp2 = img.copy()
#for (x,y,w,h) in faces:
 #cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
 
#for (x,y,w,h) in faces2:
 #cv2.rectangle(temp2,(x,y),(x+w,y+h),(255,0,0),2)
#ax[0].imshow(temp[:,:,::-1]) 
#ax[1].imshow(temp2[:,:,::-1])
#plt.show()

#주석처리된코드는 얼굴만 검출, 밑은 얼굴+눈 검출

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
faces = face_cascade.detectMultiScale(img, scaleFactor = 1.1)
faces2 = face_cascade.detectMultiScale(img, scaleFactor = 1.8)
temp = img.copy()
temp2 = img.copy()
for (x,y,w,h) in faces:
 cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
 roi_color = temp[y:y+h, x:x+w]
 eyes = eye_cascade.detectMultiScale(roi_color)
 for (ex,ey,ew,eh) in eyes:
 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
 
for (x,y,w,h) in faces2:
 cv2.rectangle(temp2,(x,y),(x+w,y+h),(255,0,0),2)
 roi_color = temp2[y:y+h, x:x+w]
 eyes = eye_cascade.detectMultiScale(roi_color)
 for (ex,ey,ew,eh) in eyes:
 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
ax[0].imshow(temp[:,:,::-1]) 
ax[1].imshow(temp2[:,:,::-1])
plt.show()

#minNeighbors
'''
객체가 있는지 판단하기 위해 사용되는 변수
예를들어 minNeighbors=3이라는 것은 3개 이상의 윈도우에서 객체가 있어야 검출한다.
즉, minNeighbors 값이 클수록 정확도는 높지만, 검출률이 낮다
'''

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,20))
faces = face_cascade.detectMultiScale(img, minNeighbors = 0)
faces2 = face_cascade.detectMultiScale(img, minNeighbors = 1)
faces3 = face_cascade.detectMultiScale(img, minNeighbors = 3)
faces4 = face_cascade.detectMultiScale(img, minNeighbors = 50)
temp = img.copy()
temp2 = img.copy()
temp3 = img.copy()
temp4 = img.copy()
for (x,y,w,h) in faces:
 cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
 
for (x,y,w,h) in faces2:
 cv2.rectangle(temp2,(x,y),(x+w,y+h),(255,0,0),2)
 
for (x,y,w,h) in faces3:
 cv2.rectangle(temp3,(x,y),(x+w,y+h),(255,0,0),2)
 
for (x,y,w,h) in faces4:
 cv2.rectangle(temp4,(x,y),(x+w,y+h),(255,0,0),2)
ax[0].imshow(temp[:,:,::-1]) 
ax[0].set_title('minNeighbors=0')
ax[1].imshow(temp2[:,:,::-1])
ax[1].set_title('minNeighbors=1')
ax[2].imshow(temp3[:,:,::-1])
ax[2].set_title('minNeighbors=3')
ax[3].imshow(temp4[:,:,::-1])
ax[3].set_title('minNeighbors=50')
plt.show()


