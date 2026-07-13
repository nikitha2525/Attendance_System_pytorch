import cv2
import json
import os
import pandas as pd
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
with open("classes.json", "r") as f:
    class_names = json.load(f)
print("Classes Loaded:")
print(class_names)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
model = torchvision.models.resnet18(
    weights=None
)
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)
model.load_state_dict(
    torch.load(
        "face.pth",
        map_location=device
    )
)
model.to(device)
model.eval()
print("Model Loaded Successfully")
def mark_attendance(name):
    file = "attendance.csv"
    today = datetime.now().strftime(
        "%Y-%m-%d"
    )
    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )
    # If file exists load it
    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(
            columns=[
                "Name",
                "Date",
                "Time",
                "Status"
            ]
        )
    # Check duplicate attendance
    already_present = (
        (df["Name"] == name) &
        (df["Date"] == today)
    ).any()
    if already_present:
        return "Present"
    else:
        new_entry = pd.DataFrame({
            "Name":[name],
            "Date":[today],
            "Time":[current_time],
            "Status":["Present"]
        })
        df = pd.concat(
            [df,new_entry],ignore_index=True
)
        df.to_csv(file,index=False
        )
        return "Present"

cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
print("Haar cascade path:", cascade_path)

if not os.path.exists(cascade_path):
    print("Haar cascade not found in OpenCV data path.")
    cascade_path = os.path.join(
        os.path.dirname(__file__),
        "haarcascade_frontalface_default.xml"
    )
    if os.path.exists(cascade_path):
        print("Using local Haar cascade:", cascade_path)
    else:
        print("Local Haar cascade also missing. Downloading from OpenCV GitHub...")
        try:
            import urllib.request
            download_url = (
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
                "haarcascade_frontalface_default.xml"
            )
            urllib.request.urlretrieve(download_url, cascade_path)
            print("Downloaded Haar cascade to:", cascade_path)
        except Exception as e:
            print("ERROR: Could not download Haar cascade:", e)
            print("Please download the file manually from:")
            print("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
            exit(1)

face_detector = cv2.CascadeClassifier(cascade_path)
if face_detector.empty():
    print("ERROR: Failed to load Haar cascade classifier from:", cascade_path)
    exit(1)

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)
cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)
if not cap.isOpened():
    print("Webcam not found")
    exit()
print("Webcam Started")
print("Press Q to Quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80,80)
    )
    for (x,y,w,h) in faces:
        # Crop face
        face = frame[
            y:y+h,
            x:x+w
        ]
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )
        image = Image.fromarray(face)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        # Prediction
        with torch.no_grad():
            output = model(image)
            confidence, prediction = torch.max(
                torch.softmax(output,1),
                1
            )
        confidence = confidence.item()
        name = class_names[
            prediction.item()
        ]
        # Confidence threshold
        if confidence > 0.70:
            status = mark_attendance(name)
        else:
            name = "Unknown"
            status = "Not Recognized"
        # Draw Rectangle
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )
        # Display Name
        cv2.putText(
            frame,
            name,
            (x,y-35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )
        # Display Status
        cv2.putText(
            frame,
            status,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )
    cv2.imshow(
        "Attendance System",
        frame
    )
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()