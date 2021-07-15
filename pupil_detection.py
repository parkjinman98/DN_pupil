import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope

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

# HSV

# 이미지 파일을 컬러로 불러옴
# cv2.imread로 불러온 이미지는 RGB Color 모델이다.
img_color = cv2.imread('image/balls.jpg')

# cvtColor 함수를 이용하여 hsv 색공간으로 변환
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# hsv 이미지에서 객체의 HSV color 범위를 지정
lower_yellow = (20, 20, 100)
upper_yellow = (32, 255, 255)

# 범위내의 픽셀들을 흰색, 나머지 검은색
img_mask_hsv = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득
img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask_hsv)

# 마스크 이미지 출력
# 범위 내에 들어가는 흰색은 255, 검은색은 0
plt.imshow(img_mask_hsv, cmap='gray')
plt.show()

#원본 이미지와 노란색공을 제외한 나머지 부분이 검은색으로 표현된 이미지 출력
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].imshow( cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")

ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].imshow( cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
ax[1].set_title("Result")
plt.show()

#동공검출에 적용

# 영상을 이미지로 바꾸기
vidcap = cv2.VideoCapture('data/samples/sample_2.mp4') success,image = vidcap.read()
count = 1 success = True
while success: 
 try:
  success,image = vidcap.read()
  cv2.imwrite("data_0702/sample/image{}.jpg".format(str(count).zfill(3)), image)
  #print("saved image %d.jpg" % count)
  count += 1
 except: 
  break
# Haar cascade를 이용해서 눈 찾기

# 원본이미지
img = cv2.imread('data_0702/sample/image041.jpg') 
plt.imshow(img[:,:,::-1])

# 얼굴 찾기
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(img)

temp = img.copy()

for (x,y,w,h) in faces: 
 cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
 
plt.imshow(img[:,:,::-1])

# 눈 찾기
eyes = eye_cascade.detectMultiScale(img) 
temp = img.copy()
for (x,y,w,h) in eyes:
 # 눈 바운딩 박스 그리기
 cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),4)

plt.imshow(temp[:,:,::-1])

# 바운딩 박스의 폭이 가장 넓은 상위 2개의 박스만 찾기
eyes = eye_cascade.detectMultiScale(img) 
eyes = eyes[eyes[:,2].argsort()[-2:]] temp = img.copy()
for (x,y,w,h) in eyes:
 # 눈 바운딩 박스 그리기
 cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),4) 

plt.imshow(temp[:,:,::-1])

#HSV Color model을 이용해 빛이 있는 쪽의 눈(target eye) 찾기

# hsv 변환
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 빛의 범위
lower_light = (0, 50, 200) 
upper_light = (255, 255, 255)

mask_light = cv2.inRange(img_hsv, lower_light, upper_light)

plt.imshow(mask_light, cmap='gray')


'''
바운딩박스의 x값이 이미지의 왼쪽에 있으면 왼쪽눈, 오른쪽에 있으면 오른쪽 눈으로 정의
바운딩박스 영역에서 빛이 비추는 비율(10%)이 일정 값 이상이고 비율이 큰쪽을 target_eye로 정의
'''
height, width = img.shape[:2]

x_l, y_l, w_l, h_l = eyes[eyes[:,0] < width//2].flatten()
x_r, y_r, w_r, h_r = eyes[eyes[:,0] > width//2].flatten()

left_value = np.sum(mask_light[y_l:y_l+h_l,x_l:x_l+w_l])/(w_l*h_l*255)
right_value = np.sum(mask_light[y_r:y_r+h_r,x_r:x_r+w_r])/(w_r*h_r*255)

temp = img.copy()
if (left_value>right_value):
   if left_value>0.1: 
       cv2.rectangle(temp,(x_l,y_l),(x_l+w_l,y_l+h_l),(0,255,0),4)
   else:
       if right_value>0.1:
           cv2.rectangle(temp,(x_r,y_r),(x_r+w_r,y_r+h_r),(0,255,0),4)
plt.imshow(temp[:,:,::-1])

#2차 회의

'''
동공 경계 검출 방법
1. Haar cascade를 이용하여 눈 검출
2. HSV color model을 이용하여 빛이 있는 눈 선택
3. Erosion & Dilation을 이용하여 눈꼬리 제거
4. Sampling을 한 후 Robust Covariance Outlier Detection을 이용해 대략적인 각막의 중심 도출
5. MCD(Minimum Covariance Determinant) center를 기준으로 RoI 결정
6. Hough circle detection을 이용해 RoI 내에 MCD 중심과 가장 가까운 원을 동공의 경계로 정함
'''

# Haar cascade를 이용해서 눈 찾기

img = cv2.imread('data_0712/sample/image001.jpg')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(img)
eyes = eyes[eyes[:,2].argsort()[-2:]]
temp = img.copy()
for (x,y,w,h) in eyes: 
    # 눈 바운딩 박스 그리기
    cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),4) 
plt.imshow(temp[:,:,::-1])

# MCD 중심 찾기
mcd_centers=[]
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,15)) 
for i, eye in enumerate(eyes):
    x,y,w,h = eye
    # 눈 이미지만 추출
    img_eye = img[y:y+h, x:x+w,:]
    ax[i,0].imshow(img_eye[:,:,::-1])
    img_gray = cv2.cvtColor(img_eye, cv2.COLOR_BGR2GRAY) 
    # img_gray = cv2.medianBlur(img_gray, 5) # 그레이스케일된 이미지 블로우처리
    # img_gray 10% threshold
    threshold = np.percentile(img_gray.flatten(), 10)
    _, binary = cv2.threshold(img_gray,threshold,255, cv2.THRESH_BINARY) 
    binary = 255-binary
    ax[i,1].imshow(binary, cmap='gray')

    binary = cv2.erode(binary, None, iterations=3)
    binary = cv2.dilate(binary, None, iterations=5) # (X,Y) 
    # binary = cv2.erode(binary, None, iterations=5) 
    ax[i,2].imshow(binary, cmap = 'gray')
    # 이진화 된 이미지에서 검은색 부분을 좌표로 바꿈
    temp=np.where(binary==0) 
    X=np.concatenate([[temp[0]], [temp[1]]]).T

    #Robust covariance, contamination:
    mcd = EllipticEnvelope(support_fraction = 0.6, contamination=0.4)

    if len(X)<3: 
        continue
    elif len(X)<300:
        y_pred = mcd.fit(X).predict(X)
    else:
        y_pred = mcd.fit(X[::100]).predict(X[::100])
  
    mcd_center = mcd.location_.astype(int) mcd_center=mcd_center[::-1]
    temp = img_eye.copy()
    cv2.circle(temp, (mcd_center[0],mcd_center[1]),3,(0,0,255),3) 
    ax[i,3].imshow(temp[:,:,::-1])
    mcd_centers.append(mcd_center+np.array([x,y]))

plt.show()

# 빛이 있는 눈의 MCD center를 중심으로 작은 RoI 찾기

img = cv2.imread('data_0712/sample/image030.jpg')
''' 
In [101]:  
 mcd_centers
Out[101]:
 [array([985, 418]), array([226, 467])]
'''
def get_target_eye(img, mcd_centers, max_radious):
    mcd_centers = np.array(mcd_centers)

    height, width = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_light = (0, 50, 200) upper_light = (255, 255, 255)
    
    mask_light = cv2.inRange(img_hsv, lower_light, upper_light)
    
    try:
        left_eye = mcd_centers[mcd_centers[:,0] < width//2].flatten()
        if len(left_eye)>2:
           left_eye=left_eye[:2]
        
        x,y = left_eye
        w = max_radious+10
        h=w
        left_value = np.sum(mask_light[max(0,y-h):min(y+h,height),max(0,x-w):min(x+w)
    except:
        # print('There is no left eye') left_eye = []
        left_value = 0
    try:
        right_eye = mcd_centers[mcd_centers[:,0] >= width//2].flatten() 
        if len(right_eye)>2:
            right_eye=right_eye[:2]
        x,y = right_eye
        w = max_radious+10
        h=w
        right_value = np.sum(mask_light[max(0,y-h):min(y+h,height),max(0,x-w):min(x+w)
    except:
        # print('There is no right eye') right_eye = []
        right_value = 0
    threshold = (max_radious+10)**2*0.1*255
    
    if (left_value < threshold) & (right_value < threshold): 
        # print('Unlighted')
        return [], 2
    if left_value < right_value: 
        return right_eye, 1
    else:
        return left_eye, 0
width, height = img.shape[:2]
max_radious = int(np.sqrt(width**2+height**2)//21)
target_eye, state = get_target_eye(img, mcd_centers, max_radious)
                                        
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))
                                        
# MCD center
x = int(target_eye[0]) 
y = int(target_eye[1])
h = max_radious + 10 
w=h
                                        
# MCD center 중심으로 눈 RoI 결정
img_eye = img[max(0,y-h):min(y+h, img.shape[0]),max(0,x-w):min(x+w,img.shape[1]),:]
temp = img.copy()
cv2.rectangle(temp, (x-w,y-h),(x+w,y+h),(255,255,0), 10)
ax[0].imshow(temp)
img_gray = cv2.cvtColor(img_eye, cv2.COLOR_BGR2GRAY)
# 이진화
threshold = np.percentile(img_gray.flatten(), 10) # 하위 10% 값을 기준으로 이진화
_, binary = cv2.threshold(img_gray,threshold,255, cv2.THRESH_BINARY) 
ax[1].imshow(binary, cmap='gray') # 이진화 결과
binary = cv2.dilate(binary, None, iterations=3)
binary = cv2.erode(binary, None, iterations=1)
ax[2].imshow(binary, cmap='gray')# 전처리 결과
                                        
# Hough Circle Detection
circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 
                           dp = 1,
                           minDist = 100, 
                           param1=200,
                           param2=10, 
                           minRadius=int(max_radious*0.1), 
                           maxRadius=int(max_radious*0.6))
try:
    circles = np.uint16(np.around(circles)) 
    circles = circles[0][0]
except:
    # print("There is no pupil") 
    pass
temp = img_eye.copy()
cv2.circle(temp, (circles[0],circles[1]),circles[2],(255,0,0),3) 
cv2.circle(temp, (h,h),2, (0,0,255),2) 
ax[3].imshow(temp[:,:,::-1])
plt.show()   
