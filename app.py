from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import subprocess
import face_recognition
import cv2
import pickle
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')

# Route to serve the main page
@app.route('/')
def index():
    # Load attendance logs from CSV
    logs = []
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        logs = [f"{row['Name']} at {row['Time']} in {row['Location']}" for _, row in df.iterrows()]
    return render_template('index.html', logs=logs)

# Route to start the attendance system
@app.route('/start_attendance')
def start_attendance():
    try:
        subprocess.run(["python", "attendance_system.py"], check=True)
        return jsonify({"message": "Attendance system started! Check attendance.csv for updates."})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})

# Route to handle student registration
@app.route('/register_student', methods=['POST'])
def register_student():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    class_name = request.form['class']
    image_data = request.form['image'].split(',')[1]  # Extract base64 data
    with open("temp.jpg", "wb") as f:
        f.write(base64.b64decode(image_data))

    # Encode the captured face
    image = face_recognition.load_image_file("temp.jpg")
    encoding = face_recognition.face_encodings(image)[0]

    # Update face_encodings.pkl
    encoding_file = "face_encodings.pkl"
    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    else:
        known_face_encodings = [encoding]
        known_face_names = [name]
    with open(encoding_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    # Save student details to CSV
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

if __name__ == "__main__":
    app.run(debug=True)