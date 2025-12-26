
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
from joblib import load
import serial
import serial.tools.list_ports   # ✅ for auto-detect ESP32
import os
print("📂 Current script location:", os.getcwd())


# ============================
# Auto-detect ESP32
# ============================
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Silicon Labs" in port.description or "CH340" in port.description:
            return port.device
    return None

esp = None
port = find_esp32_port()
if port:
    try:
        esp = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)  # wait for ESP32 to reset
        print(f"✅ Connected to ESP32 on {port}")
    except Exception as e:
        print(f"⚠️ Could not open {port}: {e}")
        esp = None
else:
    print("❌ ESP32 not found. Continuing without serial control.")

# ============================
# Constants
# ============================
DATA_DIR = "Images"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
ATTENDANCE_FILE = "Attendance.csv"
MODEL_PATH = 'helmet_svm.joblib'

# ============================
# Load Helmet Detection Model
# ============================
try:
    model = load(MODEL_PATH)
    print("✅ Helmet SVM model loaded.")
except:
    print("❌ Could not load 'helmet_svm.joblib'. Train it first.")
    exit()

# Load face detector
face_cascade = cv2.CascmadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ Haar Cascade load failed.")
    exit()


# ============================
# Register Face
# ============================
def register_face(person_name, max_images=50):
    person_dir = os.path.join(DATA_DIR, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"[INFO] Created folder: {person_dir}")
    else:
        print(f"[INFO] Folder exists. Adding new images.")

    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()
    print(f"[INFO] Capturing face images for {person_name}... (auto stops after 4 sec)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = time.time() - start_time
        if elapsed > 4:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Image {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Registering Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} images for {person_name}.")


# ============================
# Encode Faces
# ============================
def encode_faces():
    known_encodings, known_names = [], []
    print("[INFO] Encoding faces...")

    for person_name in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(person_name)

    print(f"[INFO] Encoded {len(known_encodings)} face(s).")
    return known_encodings, known_names


# ============================
# Attendance
# ============================
def mark_attendance(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Name", "Time"])

    if name not in df["Name"].values:
        new_row = pd.DataFrame([{"Name": name, "Time": now}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[ATTENDANCE] Marked {name} at {now}")


# ============================
# HOG Feature Extraction
# ============================
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog.compute(gray).flatten()


# ============================
# Combined System
# ============================
def start_combined_system():
    known_encodings, known_names = encode_faces()
    if not known_encodings:
        print("[ERROR] No known faces found. Register first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("[INFO] Starting Face Attendance + Helmet Detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face recognition (scaled down for speed)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        # Helmet detection
        features = extract_hog_features(frame).reshape(1, -1)
        helmet_pred = model.predict(features)[0]

        if helmet_pred == 1:  # Helmet worn
            helmet_label = "Helmet WORN"
            helmet_color = (0, 255, 0)
            print("Helmet Status: 1 (WORN)")
            if esp:
                esp.write(b'1')  # LED ON
        else:
            helmet_label = "Helmet NOT WORN"
            helmet_color = (0, 0, 255)
            print("Helmet Status: 0 (NOT WORN)")
            if esp:
                esp.write(b'0')  # LED OFF

        # Draw label
        cv2.putText(frame, helmet_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, helmet_color, 3)

        # Face recognition loop
        for face_encoding, face_loc in zip(face_encs, face_locs):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            if len(distances) > 0:
                best = np.argmin(distances)
                if distances[best] < 0.5:
                    name = known_names[best]
                    color = (0, 255, 0)
                    mark_attendance(name)

            y1, x2, y2, x1 = [v * 4 for v in face_loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Face Attendance & Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


# ============================
# Main Menu
# ============================
if __name__ == "__main__":
    while True:
        print("\n========= Face Attendance & Helmet Detection =========")
        print("1. Register New Person")
        print("2. Start Combined Attendance & Helmet Detection")
        print("3. Exit")
        choice = input("Select (1/2/3): ")

        if choice == "1":
            name = input("Enter person's name: ").strip()
            if name:
                register_face(name)
            else:
                print("[ERROR] Name cannot be empty.")
        elif choice == "2":
            start_combined_system()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")
