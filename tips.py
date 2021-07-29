
vid_img=[]
for i in range(1,count):
    vid_img.append(cv2.imread("processed_data/data_0725/processed_video_6_1/image{}.jpg".format(str(i).zfill(3))))
h,w,layers=vid_img[1].shape
fps = vidcap.get(cv2.CAP_PROP_FPS)
vid=cv2.VideoWriter("processed_data/data_0725/result_video/result_6_1.mp4", cv2.VideoWriter_fourcc(*'FMP4'),fps,(w, h))
for j in range(0,count-1):
    vid.write(vid_img[j])
vid.release()