import torch
import cv2
import os
import sys

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

args: list[int] = sys.argv

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(sys.argv[1])
n = 0
thickness = 5
timer = 0
color = (255, 0, 0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

if not cap.isOpened():
    print("Error opening video stream")

while True:
    flag, im0 = cap.read()
    if flag:
        showCrosshair = False
        fromCenter = False
        r = cv2.selectROI("Image", im0, fromCenter, showCrosshair)
        break

while cap.isOpened():
    ret, image_frame = cap.read()
    if ret == 1:
        rect_img = image_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        cv2.rectangle(image_frame, (int(r[0]), int(r[1])), (int(r[0] + r[2]), int(r[1] + r[3])), (255, 255, 0), 3)

        image_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = rect_img

        mask = object_detector.apply(rect_img)
        ret, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        detections = []
        if timer % 100 == 0:
            results = model(image_frame)
            cls = results.pandas().xyxy[0]['class']
            for i in range(len(cls)):
                if cls[i] == 0:

                    cnterCordin = [
                        (int(results.pandas().xyxy[0]['xmin'][i]) + int(results.pandas().xyxy[0]['xmax'][i])) / 2,
                        (int(results.pandas().xyxy[0]['ymin'][i]) + int(results.pandas().xyxy[0]['ymax'][i])) / 2]
                    radius = int(cnterCordin[1]) / 3

                    if int(r[0]) < int(cnterCordin[0]) < (int(r[0] + r[2])) and \
                            int(r[1]) < int(cnterCordin[1]) < (int(r[1] + r[3])):
                        image_frame = cv2.circle(image_frame, (int(cnterCordin[0]), int(cnterCordin[1])),
                                                 10, color, thickness)
                        print("in")
                        cv2.imwrite(f'../virtual_cordon/images/image_{n}.png', image_frame)
                        n = n + 1

        cv2.imshow("ROI", image_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cap.release()
cv2.destroyAllWindows()
