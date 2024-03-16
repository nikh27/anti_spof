import math
import time
import cv2
import cvzone
from ultralytics import YOLO
import face_recognition
import numpy as np
import csv
from datetime import datetime

confidence = 0.6

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("../models/n_version_1_50.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

# Load known faces
nikhil = face_recognition.load_image_file("WIN_20230106_12_38_01_Pro.jpg")
nikhil_encoded = face_recognition.face_encodings(nikhil)[0]
mayank = face_recognition.load_image_file("WIN_20231205_02_43_34_Pro.jpg")
mayank_encoded = face_recognition.face_encodings(mayank)[0]
amit = face_recognition.load_image_file("WIN_20231030_21_33_36_Pro.jpg")
amit_encoded = face_recognition.face_encodings(amit)[0]

known_face_encodings = [nikhil_encoded, mayank_encoded, amit_encoded]
known_face_names = ["nikhil", "mayank", "amit"]  # Change this line to store names

students = known_face_names.copy()

face_locations = []
face_encodings = []

print(students)#####


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > confidence:
                print(conf)####
                if classNames[cls] == 'real':
                    print("its reall haha")
                    color = (0, 255, 0)
                    # Face recognition
                    small_frame = cv2.resize(img[y1:y2, x1:x2], (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        if name in known_face_names:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            bottomLeftCornerText = (max(0, x1), max(35, y1))
                            fontScale = 1
                            fontColor = (0, 255, 0)
                            thickness = 2
                            lineType = 2
                            cv2.putText(img, f'{name} present', bottomLeftCornerText, font, fontScale, fontColor, thickness, lineType)
                            print("students",students)
                            if name in students:
                                students.remove(name)
                                current_time = now.strftime("%H:%M:%S")

                else:
                    color = (0, 0, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerText = (10, 10)
                    fontScale = 1
                    fontColor = (0, 0, 255)
                    thickness = 2
                    lineType = 2
                    cv2.putText(img, "You are fake", bottomLeftCornerText, font, fontScale, fontColor, thickness, lineType)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
