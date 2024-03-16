import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

cap = cv2.VideoCapture(0)

nikhil = face_recognition.load_image_file("WIN_20230106_12_38_01_Pro.jpg")
nikhil_encoded = face_recognition.face_encodings(nikhil)[0]
mayank = face_recognition.load_image_file("WIN_20231205_02_43_34_Pro.jpg")
mayank_encoded = face_recognition.face_encodings(mayank)[0]
amit = face_recognition.load_image_file("WIN_20231030_21_33_36_Pro.jpg")
amit_encoded = face_recognition.face_encodings(amit)[0]

known_face = [nikhil_encoded,mayank_encoded,amit_encoded]
known_name = ["nikhil","mayank","amit"]  # Change this line to store names

# list of expected faces

students = known_name.copy()

face_locations = []
face_encodings = []


#  get datetime

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# f = open(f"{current_date}.csv","w+",newline="")
#
# lnwriter = csv.writer(f)


while True:
    _, frame = cap.read()  # Corrected variable name
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    # recoginting faces

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face,face_encoding)
        face_distance = face_recognition.face_distance(known_face,face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_name[best_match_index]  # Corrected variable name

        # add the text if person is present
        if name in known_name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerText = (10,100)
            fontScale = 1.5
            fontColor = (255,0,0)
            thikness = 3
            linetype = 2
            cv2.putText(frame, f'{name} present', bottomLeftCornerText, font, fontScale, fontColor, thikness, linetype)
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                # lnwriter.writerow([name,current_time])

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
f.close()
