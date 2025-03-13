from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import face_recognition
import cv2
import pickle
from datetime import datetime
import base64
import smtplib
from email.mime.text import MIMEText
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load face encodings
def load_encodings():
    encoding_file = "face_encodings.pkl"
    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as f:
            return pickle.load(f)
    return [], []

known_face_encodings, known_face_names = load_encodings()

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
        logger.info(f"Email sent for {name}!")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

# Route to serve the main page
@app.route('/')
def index():
    logs = []
    if os.path.exists("attendance.csv"):
        try:
            df = pd.read_csv("attendance.csv")
            logs = [f"{row['Name']} at {row['Time']} in {row['Location']}" for _, row in df.iterrows()]
        except Exception as e:
            logger.error(f"Error reading attendance.csv for index: {e}")
    return render_template('index.html', logs=logs)

# Route to get the list of students
@app.route('/get_students', methods=['GET'])
def get_students():
    try:
        if os.path.exists("students.csv"):
            students_df = pd.read_csv("students.csv")
            students = students_df.to_dict('records')
        else:
            students = []
        return jsonify({"students": students})
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        return jsonify({"error": str(e)}), 500

# Route to get attendance records
@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    try:
        if os.path.exists("attendance.csv"):
            attendance_df = pd.read_csv("attendance.csv")
            attendance = attendance_df.to_dict('records')
            logger.info(f"Successfully loaded {len(attendance)} attendance records.")
        else:
            attendance = []
            logger.warning("attendance.csv not found, returning empty list.")
        return jsonify({"attendance": attendance})
    except Exception as e:
        logger.error(f"Error getting attendance: {e}")
        return jsonify({"error": str(e)}), 500

# Route to update student authorization
@app.route('/update_authorization', methods=['POST'])
def update_authorization():
    try:
        data = request.get_json()
        name = data['name']
        authorized = data['authorized']

        students_file = "students.csv"
        if os.path.exists(students_file):
            students_df = pd.read_csv(students_file)
            if name in students_df['Name'].values:
                students_df.loc[students_df['Name'] == name, 'Authorized'] = authorized
                students_df.to_csv(students_file, index=False)
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "message": "Student not found."})
        else:
            return jsonify({"success": False, "message": "No students registered."})
    except Exception as e:
        logger.error(f"Error updating authorization: {e}")
        return jsonify({"success": False, "message": f"Error updating authorization: {str(e)}"})

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
            students_df = pd.DataFrame(columns=['Name', 'Email', 'Phone', 'Class', 'RegistrationTime', 'Authorized'])
        new_student = pd.DataFrame([{
            'Name': name,
            'Email': email,
            'Phone': phone,
            'Class': class_name,
            'RegistrationTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Authorized': False
        }])
        students_df = pd.concat([students_df, new_student], ignore_index=True)
        students_df.to_csv(students_file, index=False)

        os.remove("temp.jpg")
        return jsonify({"success": True, "message": "Student registered successfully!"})
    except Exception as e:
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        logger.error(f"Error during registration: {e}")
        return jsonify({"success": False, "message": f"Error during registration: {str(e)}"})

# Route to handle attendance marking
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        image_data = request.form['image'].split(',')[1]
        with open("temp.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))

        image = face_recognition.load_image_file("temp.jpg")
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        logs = []
        if os.path.exists("attendance.csv"):
            attendance_df = pd.read_csv("attendance.csv")
        else:
            attendance_df = pd.DataFrame(columns=['Name', 'Time', 'Location'])

        if os.path.exists("students.csv"):
            students_df = pd.read_csv("students.csv")
        else:
            students_df = pd.DataFrame(columns=['Name', 'Authorized'])

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

                student = students_df[students_df['Name'] == name]
                if not student.empty and student['Authorized'].iloc[0]:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    location = "Office"

                    if name not in attendance_df['Name'].values:
                        new_entry = pd.DataFrame([{'Name': name, 'Time': current_time, 'Location': location}])
                        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                        attendance_df.to_csv("attendance.csv", index=False)
                        logs.append(f"{name} at {current_time} in {location}")
                        send_email(name, current_time)
                else:
                    logs.append(f"{name} is not authorized.")
            else:
                logs.append("Unknown face detected.")

        os.remove("temp.jpg")
        return jsonify({"log": logs, "status": "Scanning..." if not logs else "Attendance logged!"})
    except Exception as e:
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        logger.error(f"Error during attendance marking: {e}")
        return jsonify({"log": [], "status": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)