import cv2
import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

# Custom theme colors for a unique look
BG_COLOR = "#2E2E2E"  # Dark gray background
BUTTON_COLOR = "#4CAF50"  # Green buttons
TEXT_COLOR = "#FFFFFF"  # White text
LOG_BG = "#424242"  # Darker gray for log area

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

# Self-registration function
def self_register():
    name = simple_interface.entry_name.get()
    if not name:
        simple_interface.log_area.insert(tk.END, "Please enter a name!\n")
        return
    simple_interface.log_area.insert(tk.END, f"Registering {name}...\n")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        encoding = face_recognition.face_encodings(frame)[0]
        with open("face_encodings.pkl", "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
        with open("face_encodings.pkl", "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        simple_interface.log_area.insert(tk.END, f"Registered {name} successfully!\n")
    cap.release()
    cv2.destroyAllWindows()

# Attendance system function
def run_attendance_system():
    simple_interface.log_area.insert(tk.END, "Starting attendance system...\n")
    encoding_file = "face_encodings.pkl"
    if not os.path.exists(encoding_file):
        known_face_encodings, known_face_names = encode_faces()
        if known_face_encodings is None:
            simple_interface.log_area.insert(tk.END, "Failed to encode faces.\n")
            return
    else:
        with open(encoding_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        simple_interface.log_area.insert(tk.END, f"Loaded {len(known_face_names)} face encodings.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        simple_interface.log_area.insert(tk.END, "Error: Could not open webcam.\n")
        return

    csv_file = "attendance.csv"
    if os.path.exists(csv_file):
        attendance = pd.read_csv(csv_file)
    else:
        attendance = pd.DataFrame(columns=['Name', 'Time', 'Location'])
    simple_interface.log_area.insert(tk.END, "Attendance dataframe initialized.\n")

    ret, prev_frame = cap.read()
    if not ret:
        simple_interface.log_area.insert(tk.END, "Error: Failed to capture initial frame.\n")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            simple_interface.log_area.insert(tk.END, "Error: Failed to capture frame.\n")
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

                    if name not in attendance['Name'].values:
                        new_entry = pd.DataFrame([{'Name': name, 'Time': current_time, 'Location': location}])
                        attendance = pd.concat([attendance, new_entry], ignore_index=True)
                        attendance.to_csv(csv_file, index=False)
                        simple_interface.log_area.insert(tk.END, f"Logged: {name} at {current_time} in {location}\n")
                        send_email(name, current_time)

                else:
                    with open("unknown_faces.txt", "a") as f:
                        f.write(f"Unknown face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    simple_interface.log_area.insert(tk.END, "Unknown face detected.\n")

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            prev_frame = frame.copy()
        else:
            simple_interface.log_area.insert(tk.END, "Photo detected! Use a real face.\n")

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    simple_interface.log_area.insert(tk.END, f"Attendance saved to {csv_file}\n")

# Simple and unique interface class
class SimpleInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("400x500")

        # Title Label
        title = ttk.Label(root, text="Face Attendance", font=("Helvetica", 16, "bold"), foreground=TEXT_COLOR, background=BG_COLOR)
        title.pack(pady=10)

        # Name Entry
        ttk.Label(root, text="Name:", foreground=TEXT_COLOR, background=BG_COLOR).pack()
        self.entry_name = ttk.Entry(root)
        self.entry_name.pack(pady=5)

        # Buttons
        ttk.Button(root, text="Start Attendance", command=run_attendance_system, style="Custom.TButton").pack(pady=5)
        ttk.Button(root, text="Self-Register", command=self_register, style="Custom.TButton").pack(pady=5)

        # Log Area
        self.log_area = scrolledtext.ScrolledText(root, height=15, width=40, bg=LOG_BG, fg=TEXT_COLOR)
        self.log_area.pack(pady=10)

        # Custom style for buttons
        style = ttk.Style()
        style.configure("Custom.TButton", background=BUTTON_COLOR, foreground=TEXT_COLOR, font=("Helvetica", 10, "bold"))

    def update_log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

# Initialize and run the interface
if __name__ == "__main__":
    root = tk.Tk()
    simple_interface = SimpleInterface(root)
    root.mainloop()