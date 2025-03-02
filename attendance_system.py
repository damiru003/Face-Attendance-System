import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import smtplib

from email.mime.text import MIMEText

# Function to encode faces and save them to a file
def encode_faces(face_dir="faces/", output_file="face_encodings.pkl"):
    known_face_encodings = []
    known_face_names = []

    # Check if the faces directory exists
    if not os.path.exists(face_dir):
        print(f"Error: '{face_dir}' directory not found. Please create it and add face images.")
        return None, None

    # Load and encode each image
    for filename in os.listdir(face_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(face_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Check if a face was found
                encoding = encodings[0]  # Take the first face
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split('.')[0])  # Use filename as name
            else:
                print(f"Warning: No face found in {filename}")

    # Save encodings to a file
    if known_face_encodings:
        with open(output_file, 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print(f"Face encodings saved to {output_file}!")
    else:
        print("Error: No valid face encodings generated.")
        return None, None

    return known_face_encodings, known_face_names

# Function to send email notification (advanced feature)
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

# Main attendance system function
def run_attendance_system():
    # Step 1: Encode faces if not already done
    encoding_file = "face_encodings.pkl"
    if not os.path.exists(encoding_file):
        known_face_encodings, known_face_names = encode_faces()
        if known_face_encodings is None:
            return  # Exit if encoding failed
    else:
        # Load existing encodings
        with open(encoding_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"Loaded {len(known_face_names)} face encodings from {encoding_file}")

    # Step 2: Initialize webcam and attendance log
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load or create attendance CSV
    csv_file = "attendance.csv"
    if os.path.exists(csv_file):
        attendance = pd.read_csv(csv_file)
    else:
        attendance = pd.DataFrame(columns=['Name', 'Time'])

    # Step 3: Real-time face recognition loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Log attendance if not already logged
                if name not in attendance['Name'].values:
                    new_entry = pd.DataFrame([{'Name': name, 'Time': current_time}])
                    attendance = pd.concat([attendance, new_entry], ignore_index=True)
                    attendance.to_csv(csv_file, index=False)
                    print(f"Attendance logged for {name} at {current_time}")

                    # Send email notification (replace with your email details)
                    send_email(name, current_time, 
                               sender="your_email@gmail.com", 
                               receiver="admin_email@gmail.com", 
                               password="your_app_password")  # Use Gmail app password

            # Draw rectangle and label on frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the video feed
        cv2.imshow('Attendance System', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Attendance saved to {csv_file}")

# Run the system
if __name__ == "__main__":
    run_attendance_system()