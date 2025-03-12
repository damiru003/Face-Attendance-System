import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import numpy as np
import time

# Function to encode faces and save them to a file
def encode_faces(face_dir="faces/", output_file="face_encodings.pkl"):
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(face_dir):
        print(f"Error: '{face_dir}' directory not found.")
        return None, None

    for filename in os.listdir(face_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(face_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split('.')[0])
            else:
                print(f"Warning: No face found in {filename}")

    if known_face_encodings:
        with open(output_file, 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print(f"Face encodings saved to {output_file}!")
    else:
        print("Error: No valid face encodings generated.")
        return None, None

    return known_face_encodings, known_face_names

# Function to send email notification
def send_email(name, time, sender="your_email@gmail.com", receiver="admin_email@gmail.com", password="your_app_password"):
    try:
        msg = MIMEText(f"Attendance logged for {name} at {time}")
        msg['Subject'] = 'Attendance Update'
        msg['From'] = sender
        msg['To'] = receiver
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"Email sent for {name}!")
    except Exception as e:
        print(f"Email sending failed: {e}")

# Function to detect if it's a real face (vs. photo) using motion
def is_real_face(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    motion = cv2.countNonZero(thresh)
    return motion > 500

# Main attendance system function
def run_attendance_system():
    print("Starting attendance system...")
    encoding_file = "face_encodings.pkl"
    if not os.path.exists(encoding_file):
        known_face_encodings, known_face_names = encode_faces()
        if known_face_encodings is None:
            print("Failed to encode faces. Please register students first.")
            return
    else:
        with open(encoding_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"Loaded {len(known_face_names)} face encodings.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    csv_file = "attendance.csv"
    if os.path.exists(csv_file):
        attendance_df = pd.read_csv(csv_file)
    else:
        attendance_df = pd.DataFrame(columns=['Name', 'Time', 'Location'])
    print("Attendance dataframe initialized.")

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Failed to capture initial frame.")
        cap.release()
        return

    print("Press 'q' to quit the attendance system.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        if is_real_face(prev_frame, frame):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    location = "Office"

                    if name not in attendance_df['Name'].values:
                        new_entry = pd.DataFrame([{'Name': name, 'Time': current_time, 'Location': location}])
                        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                        attendance_df.to_csv(csv_file, index=False)
                        print(f"Logged: {name} at {current_time} in {location}")
                        send_email(name, current_time)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            prev_frame = frame.copy()
        else:
            cv2.putText(frame, "Photo detected! Use a real face.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Attendance saved to {csv_file}")

# Enhanced feature: Run attendance with a delay to allow camera setup
def run_with_delay():
    print("Waiting 2 seconds for camera setup...")
    time.sleep(2)
    run_attendance_system()

if __name__ == "__main__":
    run_with_delay()