# 📸 Day 58 — CNN-Based Smart Attendance System with Face Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![CNN](https://img.shields.io/badge/CNN-Face%20Recognition-FF6F00?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-CSV%20Logging-150458?style=flat-square&logo=pandas&logoColor=white)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2058-blueviolet?style=flat-square)

**Building a face recognition model is only part of the solution. The real challenge lies in collecting a diverse dataset, handling lighting conditions, and ensuring the model generalizes to real-world scenarios.**

</div>

---

## 📌 Overview

Traditional attendance systems require manual roll calls or ID card scans. Day 58 builds an **AI-powered alternative** — a CNN trained on student face images that detects faces via OpenCV, classifies them using PyTorch, and automatically logs attendance with a timestamp to a CSV file. Unknown faces are flagged and rejected without creating any record.

> **Hard truth learned today:** Building a face recognition model is only part of the solution. The real challenge lies in collecting a diverse dataset, handling different lighting conditions and camera angles, and ensuring the model generalizes well to real-world scenarios.

---

## 🏗️ System Architecture

```
 Image / Webcam Feed
         │
         ▼
┌─────────────────────┐
│  OpenCV Face        │  Haar Cascade / DNN detector
│  Detection          │  Crops and resizes face region → (64×64)
└────────┬────────────┘
         │  face ROI (region of interest)
         ▼
┌─────────────────────┐
│  Preprocessing      │  Resize, normalize, convert to tensor
│                     │  shape: (1, 3, 64, 64)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  PyTorch CNN        │  Conv → ReLU → Pool → Conv → ReLU → Pool
│  Classifier         │  → Flatten → FC → Softmax
└────────┬────────────┘
         │  class probabilities
         ▼
┌─────────────────────┐
│  Confidence Check   │  confidence ≥ threshold?
└────┬───────────────-┘
     │
     ├── YES → Student name identified
     │         ┌──────────────────────────────────┐
     │         │  Mark attendance in CSV          │
     │         │  Name | Date | Time | Status     │
     │         └──────────────────────────────────┘
     │
     └── NO  → "Not Recognized" displayed
               No CSV record created
```

---

## 📂 Dataset Structure

```
dataset/
├── Nikitha/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   ├── img_003.jpg
│   └── ...  (≥ 50 images recommended)
├── Student_2/
│   ├── img_001.jpg
│   └── ...
├── Student_3/
│   └── ...
└── Student_N/
    └── ...
```

Each folder name becomes the class label. `torchvision.datasets.ImageFolder` reads this structure automatically — folder name → class index → student name at inference.

---

## 🧠 CNN Architecture

```
Input: (batch, 3, 64, 64)  ← RGB face image, 64×64
  │
  ▼
Conv2d(3 → 32, kernel=3, padding=1)
ReLU
MaxPool2d(2×2)                         → (batch, 32, 32, 32)
  │
  ▼
Conv2d(32 → 64, kernel=3, padding=1)
ReLU
MaxPool2d(2×2)                         → (batch, 64, 16, 16)
  │
  ▼
Conv2d(64 → 128, kernel=3, padding=1)
ReLU
MaxPool2d(2×2)                         → (batch, 128, 8, 8)
  │
  ▼
Flatten                                → (batch, 128×8×8) = (batch, 8192)
  │
  ▼
Linear(8192 → 256)
ReLU
Dropout(0.5)
  │
  ▼
Linear(256 → num_students)             ← one neuron per student
  │
  ▼
Output: class logits → softmax → P(student_k)
```

**What each layer learns:**
- **Conv1** — low-level features: edges, skin tone, lighting gradients
- **Conv2** — mid-level: eye shapes, nose contours, jawline curves
- **Conv3** — high-level: face identity features unique to each student
- **FC layers** — combines all spatial features into a classification decision

---

## 🔬 What I Implemented

### 1. Dataset Preparation & DataLoader

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ─── Transforms ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    # ImageNet mean (good default)
        std= [0.229, 0.224, 0.225]
    )
])

# ─── Load from folder structure ──────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(root='dataset/', transform=transform)

# student name → index mapping
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Students registered : {num_classes}")
print(f"Class mapping       : {full_dataset.class_to_idx}")

# ─── Train / Val Split ───────────────────────────────────────────────────────────
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

print(f"Training images     : {train_size}")
print(f"Validation images   : {val_size}")
```

### 2. CNN Model

```python
class AttendanceCNN(nn.Module):
    def __init__(self, num_classes):
        super(AttendanceCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128,kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = AttendanceCNN(num_classes=num_classes)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. Training Loop

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 30

for epoch in range(EPOCHS):
    # ─── Train ───────────────────────────────────────────────────────────────────
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # ─── Validate ────────────────────────────────────────────────────────────────
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    scheduler.step()

    train_acc = train_correct / train_size
    val_acc   = val_correct   / val_size

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# ─── Save ────────────────────────────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names':      class_names,
    'num_classes':      num_classes
}, 'models/attendance_cnn.pth')
print("✅ Model saved → models/attendance_cnn.pth")
```

### 4. Face Recognition + Attendance Marking

```python
import cv2
import pandas as pd
from datetime import datetime
from torchvision import transforms
from PIL import Image

# ─── Load Saved Model ────────────────────────────────────────────────────────────
checkpoint  = torch.load('models/attendance_cnn.pth')
class_names = checkpoint['class_names']

model = AttendanceCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ─── OpenCV Face Detector ────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── Inference Transform ─────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CONFIDENCE_THRESHOLD = 0.80   # must be ≥ 80% confident to mark attendance
marked_today = set()          # avoid duplicate entries per session

def predict_face(face_roi):
    """Run CNN on a cropped face region and return (name, confidence)."""
    face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    tensor   = infer_transform(face_pil).unsqueeze(0)     # (1, 3, 64, 64)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    name       = class_names[pred_idx.item()]
    confidence = conf.item()
    return name, confidence

def mark_attendance(name):
    """Append one record to attendance CSV."""
    now  = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    record = pd.DataFrame([[name, date, time, 'Present']],
                           columns=['Name', 'Date', 'Time', 'Status'])

    try:
        existing = pd.read_csv('attendance/attendance_log.csv')
        updated  = pd.concat([existing, record], ignore_index=True)
    except FileNotFoundError:
        updated  = record

    updated.to_csv('attendance/attendance_log.csv', index=False)
    print(f"✅ Attendance marked → {name} | {date} {time}")

# ─── Webcam Loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        name, confidence = predict_face(face_roi)

        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{name} ({confidence:.1%})"
            color = (0, 200, 0)                           # green for recognized

            if name not in marked_today:
                mark_attendance(name)
                marked_today.add(name)
        else:
            label = "Not Recognized"
            color = (0, 0, 200)                           # red for unknown

        # Draw bounding box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Smart Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📋 Attendance CSV Output

```csv
Name,Date,Time,Status
Nikitha,2026-07-13,09:14:32,Present
Student_2,2026-07-13,09:15:07,Present
Nikitha,2026-07-13,09:17:45,Present
```

> Duplicate prevention: `marked_today` set ensures each student is logged only once per session, even if their face appears multiple times in the webcam feed.

---

## 🔄 Recognition Decision Flow

```
Face detected by OpenCV
         │
         ▼
CNN forward pass
         │
         ▼
Softmax probabilities
  [0.03, 0.87, 0.05, 0.05]
         │
    max = 0.87
         │
    ┌────┴────┐
    │         │
  ≥ 0.80   < 0.80
    │         │
    ▼         ▼
 "Nikitha"  "Not Recognized"
 Green box  Red box
 CSV logged No record
```

---

## 💡 Key Learnings

- **CNNs learn facial features directly from images** — no manual feature engineering; the network discovers edges, contours, and identity patterns through training
- **Combining PyTorch with OpenCV bridges deep learning and real-time applications** — OpenCV handles camera input and face detection; PyTorch handles classification
- **Organizing labeled image folders is essential** — `ImageFolder` turns directory structure into a clean training dataset automatically
- **Confidence thresholding prevents false positives** — classifying every face as someone is worse than rejecting uncertain predictions
- **A LR scheduler improves convergence** — halving the learning rate every 10 epochs lets the model fine-tune after initial rapid learning

---

## ⚠️ Limitations & Real-World Challenges

| Challenge | Detail |
|---|---|
| Lighting sensitivity | Model trained in one lighting condition may fail in another |
| Camera angle variance | Frontal training images may not generalize to profile/tilted faces |
| Small per-student dataset | <20 images per student leads to poor generalization |
| Glasses / masks | Accessories significantly alter face features the CNN relies on |
| Identical twins | CNN may confuse visually similar faces |
| No liveness detection | Printed photos could fool the system |

---

## 🗂️ Project Structure

```
day-58-attendance-system/
├── train.py                     # CNN training pipeline
├── recognize.py                 # Webcam loop + attendance marking
├── model.py                     # AttendanceCNN class
│
├── dataset/                     # Training images (organized by student)
│   ├── Nikitha/
│   ├── Student_2/
│   └── Student_N/
│
├── models/
│   └── attendance_cnn.pth       # Saved model + class_names
│
├── attendance/
│   └── attendance_log.csv       # Auto-generated attendance records
│
├── outputs/
│   ├── training_curves.png
│   └── sample_recognition.png
│
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-58-attendance-system
cd day-58-attendance-system
pip install -r requirements.txt

# 1. Collect face images into dataset/<StudentName>/ folders
# 2. Train the CNN
python train.py

# 3. Start real-time attendance
python recognize.py
# Press 'q' to stop
```

**Requirements:**
```
torch
torchvision
opencv-python
pandas
pillow
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 58 of 100 — Real-World CV Application: CNN Face Recognition + Automated Attendance

| ← Previous | Current | Next → |
|---|---|---|
| [Day 57 — LSTM Story Generator](#) | **Day 58 — Attendance System** | [Day 59](#) |


---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #PyTorch #CNN #FaceRecognition #OpenCV #ComputerVision #AttendanceSystem</sub>
</div>
