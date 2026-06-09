# Train YOLO Model

A custom object detection project using [Ultralytics YOLO11](https://docs.ultralytics.com/) trained to detect 3 objects — **bottle**, **car**, and **cup** — using a custom dataset labeled with [Label Studio](https://labelstud.io/). Includes a real-time webcam detection script.

---

## Project Structure

```
train-yolo-model/
├── README.md
├── data/
│   ├── images/           # Original labeled images (45 total)
│   ├── labels/           # YOLO-format annotation .txt files
│   ├── train/            # 80% split — used for training (36 images)
│   │   ├── images/
│   │   └── labels/
│   ├── validation/       # 20% split — used for evaluation (9 images)
│   │   ├── images/
│   │   └── labels/
│   ├── classes.txt       # Class names: bottle, car, cup
│   └── notes.json        # Contains files specific to Label Studio(ignorable)
├── data.yaml
├── detector.py
├── main.py
├── notebooks/
│   ├── train_yolo.ipynb
│   └── yolo11s.pt
├── pyproject.toml
├── requirements.txt
├── runs/
│   └── detect/
│       ├── predict/
│       │   ├── 09e4ea26-car12.jpg
│       │   ├── 5a26b2b4-car9.jpg
│       │   ├── 6336ee5d-car3.jpg
│       │   ├── 6f941e56-bottle9.jpg
│       │   ├── 767ee118-cup15.jpg
│       │   ├── 8d242366-cup12.jpg
│       │   ├── d7233f37-bottle6.jpg
│       │   ├── f1be179e-bottle15.jpg
│       │   └── f2ac9bee-car10.jpg
│       ├── predict-2/
│       │   ├── 09e4ea26-car12.jpg
│       │   ├── 5a26b2b4-car9.jpg
│       │   ├── 6336ee5d-car3.jpg
│       │   ├── 6f941e56-bottle9.jpg
│       │   ├── 767ee118-cup15.jpg
│       │   ├── 8d242366-cup12.jpg
│       │   ├── d7233f37-bottle6.jpg
│       │   ├── f1be179e-bottle15.jpg
│       │   └── f2ac9bee-car10.jpg
│       ├── predict-3/
│       │   ├── 09e4ea26-car12.jpg
│       │   ├── 5a26b2b4-car9.jpg
│       │   ├── 6336ee5d-car3.jpg
│       │   ├── 6f941e56-bottle9.jpg
│       │   ├── 767ee118-cup15.jpg
│       │   ├── 8d242366-cup12.jpg
│       │   ├── d7233f37-bottle6.jpg
│       │   ├── f1be179e-bottle15.jpg
│       │   └── f2ac9bee-car10.jpg
│       ├── predict-4/
│       │   ├── 09e4ea26-car12.jpg
│       │   ├── 5a26b2b4-car9.jpg
│       │   ├── 6336ee5d-car3.jpg
│       │   ├── 6f941e56-bottle9.jpg
│       │   ├── 767ee118-cup15.jpg
│       │   ├── 8d242366-cup12.jpg
│       │   ├── d7233f37-bottle6.jpg
│       │   ├── f1be179e-bottle15.jpg
│       │   └── f2ac9bee-car10.jpg
│       ├── train/
│       │   ├── BoxF1_curve.png
│       │   ├── BoxPR_curve.png
│       │   ├── BoxP_curve.png
│       │   ├── BoxR_curve.png
│       │   ├── args.yaml
│       │   ├── confusion_matrix.png
│       │   ├── confusion_matrix_normalized.png
│       │   ├── labels.jpg
│       │   ├── results.csv
│       │   ├── results.png
│       │   ├── train_batch0.jpg
│       │   ├── train_batch1.jpg
│       │   ├── train_batch150.jpg
│       │   ├── train_batch151.jpg
│       │   ├── train_batch152.jpg
│       │   ├── train_batch2.jpg
│       │   ├── val_batch0_labels.jpg
│       │   ├── val_batch0_pred.jpg
│       │   └── weights/
│       │       ├── best.pt
│       │       └── last.pt
│       └── val/
│           ├── BoxF1_curve.png
│           ├── BoxPR_curve.png
│           ├── BoxP_curve.png
│           ├── BoxR_curve.png
│           ├── confusion_matrix.png
│           ├── confusion_matrix_normalized.png
│           ├── val_batch0_labels.jpg
│           └── val_batch0_pred.jpg
├── screenshots/
│   ├── data_labelling.webm
│   └── data_structure.png
└── uv.lock
```

---

## Classes

| ID | Name   |
|----|--------|
| 0  | bottle |
| 1  | car    |
| 2  | cup    |

---

## Setup

**Prerequisites:** Python 3.12+

```bash
# Clone the repo
git clone https://github.com/ritisha2503/train-yolo-model.git

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Training

All training steps are documented and runnable in [`notebooks/train_yolo.ipynb`](notebooks/train_yolo.ipynb). The notebook covers:

1. **Gather & Label Data** — 15 images per class (45 total), labeled with Label Studio
2. **Train/Val Split** — 80/20 random split into `data/train/` and `data/validation/`
3. **Configure Training** — auto-generates `data.yaml` from `classes.txt`
4. **Train Model** — fine-tunes YOLO11s for 60 epochs at 640×640 resolution
5. **Test Model** — runs the model on validation images and displays predictions
6. **Evaluate Accuracy** — reports mAP50, mAP50-95, precision, and recall per class

### Training command (equivalent)

```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.train(data='data.yaml', epochs=60, imgsz=640)
```

### Training parameters

| Parameter | Value      | Notes                                      |
|-----------|------------|--------------------------------------------|
| Model     | yolo11s.pt | Small — good balance of speed and accuracy |
| Epochs    | 60         | Recommended for datasets under 200 images  |
| imgsz     | 640        | Standard YOLO resolution                  |
| Train     | 36 images  | 80% of dataset                            |
| Val       | 9 images   | 20% of dataset                            |

---

## Real-Time Webcam Detection

Run the trained model on your webcam:

```bash
python main.py
```

A window will open showing your webcam feed with bounding boxes drawn around detected objects. Each box shows the class name and confidence score.

**Press `q` to quit.**

### How it works

- [`detector.py`](detector.py) — `ObjectDetector` class loads `best.pt` and runs inference on individual frames
- [`main.py`](main.py) — captures frames from the webcam, passes them to the detector, draws colored boxes, and displays the result with OpenCV

### Detection colors

| Class  | Color  |
|--------|--------|
| bottle | Green  |
| car    | Orange |
| cup    | Blue   |

---

## Metrics

| Metric    | Description                                               |
|-----------|-----------------------------------------------------------|
| mAP50     | Mean Average Precision at IoU ≥ 0.50 (primary metric)    |
| mAP50-95  | Stricter average across IoU thresholds 0.50–0.95          |
| Precision | Of all predicted boxes, fraction that were correct        |
| Recall    | Of all real objects, fraction that were detected          |

---

## Results

Evaluated on the 9-image validation set using `model.val()` after 60 epochs of training on YOLO11s.

### Overall

| Metric    | Score  |
|-----------|--------|
| mAP50     | 0.995  |
| mAP50-95  | 0.985  |
| Precision | 0.955  |
| Recall    | 1.000  |

### Per Class

| Class  | mAP50 | Precision | Recall |
|--------|-------|-----------|--------|
| bottle | 0.995 | 0.897     | 1.000  |
| car    | 0.995 | 0.971     | 1.000  |
| cup    | 0.995 | 0.996     | 1.000  |

The model achieves near-perfect recall across all 3 classes, meaning it successfully detects every object in the validation set. mAP50 of 0.995 indicates excellent localisation accuracy at standard IoU threshold.

---

## Dependencies

| Package      | Purpose                              |
|--------------|--------------------------------------|
| ultralytics  | YOLO model training and inference    |
| torch        | Deep learning backend                |
| opencv-python| Webcam capture and image display     |
| pyyaml       | Parsing `data.yaml` config           |
| ipykernel    | Running the Jupyter notebook         |

---

## References

[How to Train YOLO Object Detection Models in Google Colab (YOLO26, YOLO11, YOLOv8)](https://www.youtube.com/watch?v=r0RspiLG260)