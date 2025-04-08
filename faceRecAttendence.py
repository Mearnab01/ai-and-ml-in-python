import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime
# === Load known faces dynamically ===
known_face_encodings = []
known_face_names = []

photos_path = 'photos'

for filename in os.listdir(photos_path):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(photos_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# === Copy list to students for tracking attendance ===
students = known_face_names.copy()

# === CSV file setup ===
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = open(f"{current_date}.csv", 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Name", "Time"])

# === Initialize video capture ===
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        name = "Unknown"

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                csv_writer.writerow([name, current_time])
                print(f"[LOG] {name} marked present at {current_time}")

        face_names.append(name)

    # === Display on screen ===
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Attendance recording stopped by user.")
        break

# === Cleanup ===
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()
print("[INFO] Attendance recorded in CSV file.")