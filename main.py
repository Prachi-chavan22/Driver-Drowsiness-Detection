import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pyttsx3
import threading
import time
import speech_recognition as sr

# ---------------- VOICE ENGINE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 180)

voice_busy = False
last_spoken_time = 0
COOLDOWN = 8   # seconds

def speak(text):
    global voice_busy
    if voice_busy:
        return

    voice_busy = True
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

    voice_busy = False

# ---------------- SPEECH RECOGNITION ----------------
recognizer = sr.Recognizer()

def listen_user():
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=3)

        text = recognizer.recognize_google(audio)
        print("User:", text)
        return text.lower()

    except:
        return ""

# ---------------- FUNCTIONS ----------------
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth):
    A = dist.euclidean(mouth[2], mouth[3])
    B = dist.euclidean(mouth[4], mouth[5])
    C = dist.euclidean(mouth[0], mouth[1])
    return (A + B) / (2.0 * C)

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [61,291,13,14,78,308]

EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.65
FRAME_THRESHOLD = 12

eye_counter = 0
yawn_counter = 0
calibrated_pitch = None

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NORMAL"
    alert = False
    pitch = 0

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            # -------- EYES --------
            left_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in LEFT_EYE]
            right_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in RIGHT_EYE]

            ear = (calculate_EAR(left_eye)+calculate_EAR(right_eye))/2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if ear < EAR_THRESHOLD:
                eye_counter += 1
            else:
                eye_counter = 0

            if eye_counter > FRAME_THRESHOLD:
                status = "DROWSY EYES"
                alert = True

            # -------- MOUTH --------
            mouth = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in MOUTH]
            mar = calculate_MAR(mouth)

            cv2.putText(frame, f"MAR: {mar:.2f}", (10,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if yawn_counter > FRAME_THRESHOLD:
                status = "YAWNING"
                alert = True

            # -------- HEAD --------
            image_points = np.array([
                (face.landmark[1].x*w, face.landmark[1].y*h),
                (face.landmark[152].x*w, face.landmark[152].y*h),
                (face.landmark[33].x*w, face.landmark[33].y*h),
                (face.landmark[263].x*w, face.landmark[263].y*h),
                (face.landmark[61].x*w, face.landmark[61].y*h),
                (face.landmark[291].x*w, face.landmark[291].y*h)
            ], dtype="double")

            model_points = np.array([
                (0,0,0),(0,-63.6,-12.5),
                (-43.3,32.7,-26),(43.3,32.7,-26),
                (-28.9,-28.9,-24.1),(28.9,-28.9,-24.1)
            ])

            focal_length = w
            center = (w/2,h/2)

            camera_matrix = np.array([
                [focal_length,0,center[0]],
                [0,focal_length,center[1]],
                [0,0,1]
            ])

            dist_coeffs = np.zeros((4,1))

            _, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat,_ = cv2.Rodrigues(rot_vec)
            angles,_,_,_,_,_ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0]

            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if calibrated_pitch is not None:
                if pitch - calibrated_pitch > 10:
                    status = "HEAD DOWN"
                    alert = True

    # -------- ALERT SYSTEM --------
    if alert:
        cv2.rectangle(frame, (0,0), (w,70), (0,0,255), -1)
        cv2.putText(frame, status, (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        current_time = time.time()

        if current_time - last_spoken_time > COOLDOWN:

            # Ask user
            threading.Thread(target=speak,
                             args=("Are you tired?",),
                             daemon=True).start()

            time.sleep(2)

            response = listen_user()

            if any(word in response for word in ["yes","tired","sleepy","haan"]):
                threading.Thread(target=speak,
                                 args=("Wake up immediately",),
                                 daemon=True).start()
            else:
                threading.Thread(target=speak,
                                 args=("Stay alert",),
                                 daemon=True).start()

            last_spoken_time = current_time

    else:
        cv2.rectangle(frame, (0,0), (w,70), (0,255,0), -1)
        cv2.putText(frame, "NORMAL", (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

    # -------- CONTROLS --------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        calibrated_pitch = pitch
        print("✅ Head calibrated")

    if key == ord('q'):
        break

    cv2.imshow("Driver Drowsiness AI System", frame)

cap.release()
cv2.destroyAllWindows()