from ultralytics import YOLO
import cv2
import cvzone
import math
import time

model = YOLO('../models/yolov8l.pt')

# classNames = []