import cv2
from pathlib import Path
from detector import ObjectDetector

MODEL_PATH = Path(__file__).parent / 'runs/detect/train/weights/best.pt'

COLORS = {
    'bottle': (0, 255, 0),    # green
    'car':    (255, 128, 0),  # orange
    'cup':    (0, 128, 255),  # blue
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(frame, result):
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_name = result.names[int(box.cls[0])]
        label = f'{cls_name}  {conf:.0%}'
        color = COLORS.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Background rectangle behind the label text so it's readable
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return frame


def main():
    detector = ObjectDetector(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Error: could not open webcam.')
        return

    print("Webcam running — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: failed to read frame.')
            break

        result = detector.detect(frame)
        frame = draw_detections(frame, result)

        cv2.imshow('YOLO Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
