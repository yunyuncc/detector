import cv2
import numpy as np

cap = cv2.VideoCapture("/dev/video0")
cnt = 0
while(True):
    cnt = cnt+1
    name = "img/{}.jpg".format(cnt)
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.imwrite(name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()