from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone
from .models import *
import face_recognition
import cv2
import os
import csv
import numpy as np
from datetime import datetime, date
import time
import math
import cvzone
from ultralytics import YOLO

fake_face = False

#### If these directories don't exist, create them
if not os.path.isdir('faces'):
    os.makedirs('faces')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('faces'))


def load_known_faces_and_names():
    # Directory where your images are stored
    image_directory = "faces/"

    # Initialize lists to store face encodings and names
    known_faces = []
    known_names = []

    # Load images and encode faces
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = filename.split("_")[0]
            image_path = os.path.join(image_directory, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)

    return known_faces, known_names


# Load known faces and names during initialization
known_faces, known_names = load_known_faces_and_names()
known_face_encodings = known_faces
known_face_names = known_names

# Initialize list of expected faces
students = known_names.copy()

print("students", students)


def home(request):
    return render(request, 'home.html')


def main(request):
    # Retrieve attendance data directly from the Attendance model
    attendance_records = Attendance.objects.all()
    context = {
        'attendance': attendance_records,
        'total_registered': totalreg(),
        'datetoday2': date.today().strftime("%d-%B-%Y"),
        'mess': "WELCOME TO UIET"  # Replace "MESSAGE" with the message you want to pass to the template
    }
    return render(request, 'main.html', context)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_blurry_faces(frame, face_locations):
    blurry_faces = []
    for loc in face_locations:
        face_image = frame[loc[0]:loc[2], loc[3]:loc[1]]
        grayscale_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = variance_of_laplacian(grayscale_image)
        blurry_faces.append(blur_score)
    return blurry_faces

confidence = 0.6


def start(request):
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)
    model = YOLO("m_version_1_149.pt")

    classNames = ["fake", "real"]

    prev_frame_time = 0
    new_frame_time = 0

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True, verbose=False)
        print("students ->",students)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > confidence:
                    print(conf)  ####
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
                                bottomLeftCornerText = (20, 20)
                                fontScale = 1
                                fontColor = (0, 255, 0)
                                thickness = 2
                                lineType = 2
                                cvzone.putTextRect(img, "real photo",
                                                  (max(0, x1), max(35, y1)), scale=1, thickness=2, colorR=color,
                                                  colorB=color)
                                cv2.putText(img, f'{name} present', bottomLeftCornerText, font, fontScale, fontColor,
                                            thickness, lineType)
                                if name in students:
                                    students.remove(name)
                                    current_time = now.strftime("%H:%M:%S")
                                    user = User.objects.get(name=name)
                                    attendance_record = Attendance.objects.create(user=user)
                                # else:
                                #     cv2.putText(img, f'attendance done', (20, 40), font, fontScale,
                                #                 fontColor,
                                #                 thickness, lineType)


                    else:
                        color = (0, 0, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerText = (20, 20)
                        fontScale = 1
                        fontColor = (0, 0, 255)
                        thickness = 2
                        lineType = 2
                        cv2.putText(img, "it's a fake face", bottomLeftCornerText, font, fontScale, fontColor, thickness,
                                    lineType)
                        cvzone.putTextRect(img, "fake photo",
                                           (max(0, x1), max(35, y1)), scale=1, thickness=2, colorR=color,
                                           colorB=color)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    attendance_records = Attendance.objects.all()

    context = {
        'attendance': attendance_records,
        'total_registered': totalreg(),
        'datetoday2': date.today().strftime("%d-%B-%Y"),
        'mess': "WELCOME TO UIET"  # Replace "MESSAGE" with the message you want to pass to the template
    }
    return render(request, 'main.html', context)


def add(request):
    if request.method == 'POST':
        newusername = request.POST.get('newusername', '')  # Get the new username from the POST request
        newuserid = request.POST.get('newuserid', '')  # Get the new user ID from the POST request

        # Create a new User instance and save it to the database
        new_user = User(name=newusername, roll=newuserid)
        new_user.save()

        # Store the user's image
        user_image_folder = f'faces'
        if not os.path.exists(user_image_folder):
            os.makedirs(user_image_folder)

        width = 360
        height = 480
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (int(height * 4 / 3), height))
            cropped_frame = resized_frame[:,
                            int((int(height * 4 / 3) - width) / 2):int((int(height * 4 / 3) + width) / 2)]
            image_path = os.path.join(user_image_folder, f'{newusername}_{newuserid}.jpg')
            cv2.imwrite(image_path, cropped_frame)

        cap.release()
        cv2.destroyAllWindows()

        # Reload known faces and names after adding a new image
        global known_faces, known_names, known_face_encodings, known_face_names
        known_faces, known_names = load_known_faces_and_names()
        known_face_encodings = known_faces
        known_face_names = known_names

        attendance_records = Attendance.objects.all()

        context = {
            'attendance': attendance_records,
            'total_registered': totalreg(),
            'datetoday2': date.today().strftime("%d-%B-%Y"),
            'mess': "WELCOME TO UIET"  # Replace "MESSAGE" with the message you want to pass to the template
        }
        return render(request, 'main.html', context)
    else:
        return HttpResponse("Only POST requests are allowed for this view.")
