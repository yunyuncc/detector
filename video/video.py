import cv2
import numpy as np

#图片存储路径
prefix_path = "img"
#摄像头地址
dev_path = "/dev/video0"



cap = cv2.VideoCapture(dev_path)
cnt = 0
while(True):
    name = "{}/{}.jpg".format(prefix_path, cnt)
    cnt = cnt + 1
    ret, frame = cap.read()
    if ret == False:
        print("read frame fail")
        break
    cv2.imshow("frame", frame)
    cv2.imwrite(name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()