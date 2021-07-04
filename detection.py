import numpy as np
import cv2
from matplotlib import pyplot as plt


'''
입력받은 이미지 파일에서 눈을 찾는 함수
Input:
    filename: 절대경로를 포함한 파일명(str타입 입력)
    
'''


class detection():
    
    def __init__(self, filename):
        #이미지 파일 지정
        self.file=filename 

    def faceNeye(self,scale=1.1,nbhd=3):
        '''
        얼굴&눈 탐지
        scale: scaling factor
        nbhd: 객체가 있는지 판단하기 위해 사용되는 변수. 예를들어 nbhd=3이라는 것은 3개 이상의 윈도우에서 객체가 있어야 검출한다.
        '''
    
        #얼굴분류기, 눈 분류기 불러오기
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
        # 이미지 파일 불러오기
        target_img = cv2.imread(self.file)
    
        # 이미지에서 얼굴 찾기+Scaling+Neighborhood        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))
        faces = face_cascade.detectMultiScale(image = target_img, scaleFactor = scale, minNeighbors = nbhd)
        temp = target_img.copy()
        for (x,y,w,h) in faces:
            cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = temp[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        plt.show()
