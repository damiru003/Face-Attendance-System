from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import face_recognition
import cv2
import pickle
from datetime import datetime
import base64
import threading
import time

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load face encodings
def load_encodings():
    encoding_file = "face_encodings.pkl"
    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as f:
            return pickle.load(f)
    return [], []

known_face_encodings, known_face_names = load_encodings()

# Route to serve the main page
@app.route('/')
def index():
    logs = []
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        logs = [f"{row['Name']} at {row['Time']} in {row['Location']}" for _, row in df.iterrows()]
    return render_template('index.html', logs=logs)

# Route to handle student registration
@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        class_name = request.form['class']
        image_data = request.form['image'].split(',')[1]

        with open("temp.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))

        image = face_recognition.load_image_file("temp.jpg")
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            os.remove("temp.jpg")
            return jsonify({"success": False, "message": "No face detected in the image. Please try again."})
        encoding = encodings[0]

        global known_face_encodings, known_face_names
        known_face_encodings.append(encoding)
        known_face_names.append(name)
        with open("face_encodings.pkl", 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)

        students_file = "students.csv"
        if os.path.exists(students_file):
            students_df = pd.read_csv(students_file)
        else:
            students_df = pd.DataFrame(columns=['Name', 'Email', 'Phone', 'Class', 'RegistrationTime'])
        new_student = pd.DataFrame([{
            'Name': name,
            'Email': email,
            'Phone': phone,
            'Class': class_name,
            'RegistrationTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        students_df = pd.concat([students_df, new_student], ignore_index=True)
        students_df.to_csv(students_file, index=False)

        os.remove("temp.jpg")
        return jsonify({"success": True, "message": "Student registered successfully!"})
    except Exception as e:
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        return jsonify({"success": False, "message": f"Error during registration: {str(e)}"})

# Route to handle attendance marking
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        image_data = request.form['image'].split(',')[1]
        with open("temp.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))

        image = face_recognition.load_image_file("temp.jpg")
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert for OpenCV compatibility
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        logs = []
        status = "Scanning..."
        if os.path.exists("attendance.csv"):
            attendance_df = pd.read_csv("attendance.csv")
        else:
            attendance_df = pd.DataFrame(columns=['Name', 'Time', 'Location'])

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
                    attendance_df.to_csv("attendance.csv", index=False)
                    logs.append(f"{name} at {current_time} in {location}")
            else:
                logs.append("Unknown face detected.")

        os.remove("temp.jpg")
        return jsonify({"log": logs, "status": status if not logs else "Attendance logged!"})
    except Exception as e:
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        return jsonify({"log": [], "status": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)