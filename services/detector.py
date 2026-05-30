from ultralytics import YOLO
import cv2
import cvzone

# ---------------- LOAD MODEL ----------------
model = YOLO("model/ppe.pt")

# ---------------- CLASSES ----------------
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 
              'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 
              'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 
              'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 
              'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 
              'wheel loader'] 

# Used only for webcam tracking
counted_ids = set()


def reset_tracking():
    global counted_ids
    counted_ids = set()


def process_frame(img, stats, mode="image"):
    global counted_ids

    # ======================================
    # VIDEO / WEBCAM = TRACKING UNIQUE IDS
    # ======================================
    if mode in ["webcam", "video"]:

        results = model.track(img, persist=True,conf= 0.5)

        for r in results:
            for box in r.boxes:

                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                track_id = int(box.id[0])

                label = classNames[cls]

                if conf > 0.5:

                    # Count unique tracked object once
                    unique_key = f"{label}_{track_id}"

                    if unique_key not in counted_ids:
                        stats[label] = stats.get(label, 0) + 1
                        counted_ids.add(unique_key)

                    color = (0,255,0) if "NO" not in label else (0,0,255)

                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

                    cvzone.putTextRect(
                        img,
                        f"{label} ID:{track_id}",
                        (x1, y1-10),
                        scale=1,
                        thickness=2
                    )

    # ======================================
    # IMAGE = COUNT EVERY DETECTION
    # ======================================
    else:

        results = model(img)

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = classNames[cls]

                if conf > 0.5:

                    # Count every object
                    stats[label] = stats.get(label, 0) + 1

                    color = (0,255,0) if "NO" not in label else (0,0,255)

                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

                    cvzone.putTextRect(
                        img,
                        f"{label} {int(conf*100)}%",
                        (x1, y1-10),
                        scale=1,
                        thickness=2
                    )

    return img, stats