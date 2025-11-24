import cv2
import torch
from ultralytics import YOLO

WEIGHTS_PATH = "runs/detect/emotion_yolo122/weights/best.pt"   # замените на свой путь к best.pt


def clamp(val, min_v, max_v):
    return max(min_v, min(val, max_v))


def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(WEIGHTS_PATH)
    model.to(device)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]  

        results = model.predict(frame, conf=0.55)

        res = results[0]

        annotated_frame = frame.copy()

        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            names = res.names

            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                label = names.get(int(cls_id), str(int(cls_id)))
                text = f"{label} {score:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                tx1 = x1
                ty1 = y1 - th - 6
                tx2 = x1 + tw
                ty2 = y1

                if ty1 < 0:
                    ty1 = y2
                    ty2 = y2 + th + 6

                tx1 = clamp(tx1, 0, w - tw)
                tx2 = clamp(tx2, th, w)

                ty1 = clamp(ty1, 0, h - th)
                ty2 = clamp(ty2, th, h)

                cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), -1)

                cv2.putText(
                    annotated_frame,
                    text,
                    (tx1, ty2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        cv2.imshow("YOLO Emotion Detection (press ESC to quit)", annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC для того, чтобы закрыть окно
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
