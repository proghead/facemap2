# main.py
from flask import Flask, Response, render_template, request
import cv2
import mediapipe as mp
import os

app = Flask(__name__)

# Folder Setup
UPLOAD_FOLDER = "static/uploads"
FACES_FOLDER = "static/faces"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure uploads folder exists
os.makedirs(FACES_FOLDER, exist_ok=True)  # Ensure faces folder exists

# Variables for dynamic features
current_mode = "face_detection"
current_filter_index = 0

# Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# List of filter file names
filter_files = [f"filter{i}.png" for i in range(10)]  # filter0.png to filter9.png

@app.route("/")
def index():
    # Render the main dashboard
    return render_template("dashboard.html", mode=current_mode, current_filter_index=current_filter_index)

@app.route("/set_mode/<mode>")
def set_mode(mode):
    global current_mode
    current_mode = mode
    return ("", 204)  # Return an empty response for JavaScript handling

@app.route("/set_filter/<direction>")
def set_filter(direction):
    global current_filter_index
    if direction == "next":
        current_filter_index = (current_filter_index + 1) % len(filter_files)
    elif direction == "prev":
        current_filter_index = (current_filter_index - 1) % len(filter_files)
    return ("", 204)  # Return an empty response for JavaScript handling

@app.route("/video_feed")
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Access the default webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the selected mode
            if current_mode == "face_detection":
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            elif current_mode == "ar_overlay":
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                overlay = cv2.imread(os.path.join("static", "filters", filter_files[current_filter_index]), -1)  # AR Overlay
                if overlay is None:
                    print(f"Failed to load filter: {filter_files[current_filter_index]}")
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    resized_overlay = cv2.resize(overlay, (w, h))
                    frame = overlay_on_face(frame, resized_overlay, x, y)

            elif current_mode == "save_faces":
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for i, (x, y, w, h) in enumerate(faces):
                    face = frame[y:y + h, x:x + w]
                    cv2.imwrite(os.path.join(FACES_FOLDER, f'face_{i}.jpg'), face)

            elif current_mode == "deep_learning_detection":
                results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Helper function for overlay
def overlay_on_face(frame, overlay, x, y):
    h, w, _ = overlay.shape
    if overlay.shape[2] == 3:  # Add alpha channel if missing
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):  # RGB Channels
        frame[y:y + h, x:x + w, c] = (alpha_s * overlay[:, :, c] +
                                      alpha_l * frame[y:y + h, x:x + w, c])
    return frame

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
