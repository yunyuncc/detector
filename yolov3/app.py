import cv2
import numpy as np
import torch
from detector import detector
camera_path = "/dev/video0"

img_width = 640
img_height = 480

confidence = 0.85
nms_thesh = 0.6
cfgfile = "cfg/yolov3.cfg"
weightsfile = "weights/yolov3.weights"
classes_file = "data/coco.names"
CUDA = torch.cuda.is_available()
det = detector(cfgfile, weightsfile, classes_file, CUDA, batch_size = 1)
video = cv2.VideoCapture(camera_path)
while(True):
    success, frame = video.read()
    if not success:
        print("read frame fail")
        break
    detect_ress = det.detect([frame])
    for _, detect_res in enumerate(detect_ress):
        top_left_one = detect_res["top_left"]
        bottom_right_one = detect_res["bottom_right"]
        top_left = (int(top_left_one[0]*img_width), int(top_left_one[1]*img_height))
        bottom_right = (int(bottom_right_one[0]*img_width), int(bottom_right_one[1]*img_height))

        obj_score = detect_res["obj_score"]
        class_score = detect_res["class_score"]
        class_label = detect_res["class_label"]
        batch_idx = detect_res["batch_idx"]

        color = (0,255,0)
        cv2.rectangle(frame, top_left, bottom_right, color, 1)
        info = "[{}({:.3f})]obj:{:.3f}".format(class_label, class_score, obj_score)
        cv2.putText(frame,info, top_left,cv2.FONT_HERSHEY_PLAIN,1,color, 1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video.release()
cv2.destroyAllWindows()
