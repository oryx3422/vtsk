from ultralytics import YOLO
import torch

WEIGHTS_PATH = "yolo12n.pt"


def main() -> None:
    runner = YOLO(WEIGHTS_PATH)

    runner.train(
        data="Emotion-2/data.yaml",
        epochs=10,
        imgsz=512,
        batch=80,
        device = 0 if torch.cuda.is_available() else "cpu",
        workers=1,
        name="emotion_yolo12",
        cache=True,  # False, если оперативной памяти < 16GB
        save_period=10,      
    )

if __name__ == "__main__":
    main()

