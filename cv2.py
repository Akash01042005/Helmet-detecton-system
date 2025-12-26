import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
from joblib import load

# Constants
DATA_DIR = "Images"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
ATTENDANCE_FILE = "Attendance.csv"

# Load trained SVM model for helmet detection
model_path = 'helmet_svm.joblib'
try:
    model = load(model_path)
    print("✅ Helmet SVM model loaded.")
except:
    print("❌ Could not load 'helmet_svm.joblib'. Train it first.")
    exit()

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ Could not load Haar cascade for face detection.")
    exit()

# -----------------------------------------
# Register new faces
def register_face(person_name, max_images=50):
    person_dir = os.path.join(DATA_DIR, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"[INFO] Created folder: {person_dir}")
    else:
        print(f"[INFO] Folder already exists. New images will be added.")

    face_cascade_local = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    print(f"[INFO] Starting face capture for {person_name}... Auto stops after 4 seconds.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed > 4:
            print("[INFO] Time limit reached. Stopping capture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_local.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            count += 1
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Image {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Registering Face (Auto closes after 4 sec)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} face images for {person_name}.")

# -----------------------------------------
# Encode faces
def encode_faces():
    known_encodings = []
    known_names = []

    print("[INFO] Encoding face images...")
    for person_name in os.listdir(DATA_DIR):
        person_folder = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        for file_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Couldn't read {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
            else:
                print(f"[WARNING] No face found in: {img_path}")

    print(f"[INFO] Encoded {len(known_encodings)} face(s).")
    return known_encodings, known_names

# -----------------------------------------
# Attendance marking
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])

    if name not in df["Name"].values:
        new_row = pd.DataFrame([{"Name": name, "Time": dt_string}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[ATTENDANCE] Marked {name} at {dt_string}")

# -----------------------------------------
# Helmet detection helpers
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
    features = hog.compute(gray)
    return features.flatten()

# -----------------------------------------
# Combined attendance + helmet detection system
def start_combined_system(auto_close_after_detect=True):
    known_encodings, known_names = encode_faces()

    if not known_encodings:
        print("[ERROR] No face encodings found. Please register someone first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("[INFO] Starting combined Face Attendance & Helmet Detection.")
    detected_names = set()
    auto_close_delay = 3  # seconds

    last_known_detected_time = None
    last_unknown_detected_time = None

    helmet_detection_timeout = 4
    stable_helmet_start = None
    last_helmet_pred = None
    no_face_start_time = None
    last_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Resize for face_recognition speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Face recognition
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Detect faces via Haar cascade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_haar = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Helmet detection + face detection status
        if len(faces_haar) > 0:
            no_face_start_time = None  # Reset timer

            for (x, y, w, h) in faces_haar:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            features = extract_hog_features(frame).reshape(1, -1)
            helmet_pred = model.predict(features)[0]

            # ✅ Added numeric print output
            if helmet_pred == 0:
                helmet_label = "Helmet is worn"
                helmet_color = (0, 255, 0)
                print("Helmet Status: 0 (Helmet is worn)")
            else:
                helmet_label = "Helmet is NOT worn"
                helmet_color = (0, 0, 255)
                print("Helmet Status: 1 (Helmet is NOT worn)")

            # Stability logic
            if helmet_pred == last_helmet_pred:
                if stable_helmet_start is None:
                    stable_helmet_start = time.time()
                else:
                    elapsed = time.time() - stable_helmet_start
                    if elapsed >= helmet_detection_timeout:
                        print(f"⏰ Helmet prediction '{helmet_label}' stable for {helmet_detection_timeout} seconds. Exiting...")
                        break
            else:
                stable_helmet_start = None

            last_helmet_pred = helmet_pred

        else:
            helmet_label = "No face detected"
            helmet_color = (0, 255, 255)
            stable_helmet_start = None
            last_helmet_pred = None

            if no_face_start_time is None:
                no_face_start_time = time.time()
            else:
                elapsed = time.time() - no_face_start_time
                if elapsed >= helmet_detection_timeout:
                    print(f"⏰ No face detected for {helmet_detection_timeout} seconds. Exiting...")
                    break

        # Face recognition attendance logic
        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            threshold = 0.5
            name = "Unknown"
            color = (0, 0, 255)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < threshold:
                    name = known_names[best_match_index]
                    color = (0, 255, 0)

                    if name not in detected_names:
                        mark_attendance(name)
                        detected_names.add(name)
                        last_known_detected_time = time.time()
                        print(f"[INFO] Detected {name}. Auto closing webcam soon...")
                else:
                    last_unknown_detected_time = time.time()
            else:
                last_unknown_detected_time = time.time()

            # Draw name on frame
            y1, x2, y2, x1 = [val * 4 for val in face_location]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display helmet status
        cv2.putText(frame, helmet_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, helmet_color, 3)
        cv2.imshow("Face Attendance & Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quitting by user command.")
            break

        if auto_close_after_detect and last_known_detected_time:
            if time.time() - last_known_detected_time > auto_close_delay:
                detected_names_str = ", ".join(detected_names)
                print(f"[INFO] Detected {detected_names_str}. Auto closing webcam.")
                break

        if auto_close_after_detect and last_unknown_detected_time:
            if time.time() - last_unknown_detected_time > auto_close_delay:
                print("[INFO] Unknown face detected. Auto closing webcam.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Attendance and helmet detection session ended.")

# -----------------------------------------
# Main menu
if __name__ == "__main__":
    while True:
        print("\n========= Face Attendance & Helmet Detection System =========")
        print("1. Register New Person")
        print("2. Start Combined Attendance & Helmet Detection")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ")

        if choice == "1":
            name = input("Enter the person's name: ").strip()
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
            print("Invalid choice. Please select 1, 2, or 3.")
