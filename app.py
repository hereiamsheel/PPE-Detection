from flask import Flask, render_template, request, Response, redirect, url_for, send_from_directory
import cv2
import os
import time
from datetime import datetime
from services.detector import process_frame, reset_tracking

app = Flask(__name__)

# =====================================================
# ABSOLUTE PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
7
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# GLOBALS
# =====================================================
stats = {}
camera_running = False
last_output = None


# =====================================================
# HOME
# =====================================================
@app.route('/')
def index():
    return render_template("index.html")


# =====================================================
# DOWNLOAD
# =====================================================
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


# =====================================================
# UPLOAD
# =====================================================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global stats, last_output, camera_running

    reset_tracking()
    stats = {}
    camera_running = False
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == "":
            return redirect(url_for('upload'))

        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        ext = filename.split('.')[-1].lower()

        # =============================================
        # IMAGE
        # =============================================
        if ext in ['jpg', 'jpeg', 'png']:

            img = cv2.imread(path)

            if img is None:
                return "Failed to read image"

            img, stats = process_frame(img, stats, mode="image")

            filename_out = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            out_path = os.path.join(OUTPUT_FOLDER, filename_out)
            saved = cv2.imwrite(out_path, img)
            print("Saving image:", out_path)
            print("Saved:", saved)

            last_output = filename_out

        # =============================================
        # VIDEO
        # =============================================
        else:
            cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                return "Failed to open uploaded video"

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0 or fps > 120:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width == 0 or height == 0:
                return "Invalid video dimensions"

            filename_out = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, filename_out)

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            if not out.isOpened():
                return "VideoWriter failed"

            frame_count = 0
            reset_tracking()

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                frame, stats = process_frame(frame, stats, mode="video")
                frame = cv2.resize(frame, (width, height))
                out.write(frame)

            cap.release()
            out.release()

            print("Video Saved:", out_path)
            print("Frames:", frame_count)

            last_output = filename_out

        return redirect(url_for('analytics'))

    return render_template("upload.html")


# =====================================================
# LIVE PAGE
# =====================================================
@app.route('/live')
def live():
    return render_template("live.html")


# =====================================================
# START CAMERA
# =====================================================
@app.route('/start_camera')
def start_camera():
    global camera_running, stats

    reset_tracking()
    stats = {}
    camera_running = True
    return "Camera Started"


# =====================================================
# STOP CAMERA
# =====================================================
@app.route('/stop_camera')
def stop_camera():
    global camera_running
    camera_running = False
    return redirect(url_for('analytics'))


# =====================================================
# WEBCAM STREAM
# =====================================================
def generate_frames():
    global camera_running, stats, last_output

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera failed")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 7

    live_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = os.path.join(OUTPUT_FOLDER, live_name)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("VideoWriter failed")
        return

    start_time = time.time()

    while camera_running:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame,1)
        # PPE Detection
        frame, stats = process_frame(frame, stats, mode="webcam")
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        cv2.putText(
            frame,
            current_time,
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 255, 255),
            2
        )

        # ===============================
        # Duration Timer
        # ===============================
        elapsed = int(time.time() - start_time)

        mins = elapsed // 60
        secs = elapsed % 60

        timer = f"REC {mins:02}:{secs:02}"

        cv2.putText(
            frame,
            timer,
            (10, 65),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 255),
            2
        )

        # Save frame to MP4
        out.write(frame)

        # Preview image
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, "live.jpg"), frame)
        last_output = live_name
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        time.sleep(0.03)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

    cap.release()
    out.release()

    print("✅ Recording saved")


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# =====================================================
# ANALYTICS
# =====================================================
@app.route('/analytics')
def analytics():
    return render_template(
        "analytics.html",
        stats=stats,
        output=last_output
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)