import cv2
import numpy as np
import face_recognition
import pickle
import csv
from datetime import datetime

# Load encodings
with open("known_encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize attendance log
def mark_attendance(name):
    with open("attendance.csv", "r+") as f:
        lines = f.readlines()
        logged_names = [line.split(",")[0] for line in lines]
        if name not in logged_names:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            f.write(f"{name},{time_str}\n")
            print(f"[LOG] Attendance marked for {name} at {time_str}")

# Open webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(rgb_small_frame)
    encodes_cur_frame = face_recognition.face_encodings(rgb_small_frame, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(known_face_encodings, encode_face)
        face_distances = face_recognition.face_distance(known_face_encodings, encode_face)

        match_index = np.argmin(face_distances)
        name = "Unknown"

        if matches[match_index]:
            name = known_face_names[match_index]

        # Scale face location back to original frame size
        y1, x2, y2, x1 = face_loc
        y1 *= 4
        x2 *= 4
        y2 *= 4
        x1 *= 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if name != "Unknown":
            mark_attendance(name)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

