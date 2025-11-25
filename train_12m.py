from ultralytics import YOLO
import torch

def main():
    MODEL_PATH = "models/yolo12m.pt"

    model = YOLO(MODEL_PATH)

    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data="Emotion-2/data.yaml",

        epochs=10,
        imgsz=640,
        batch=15,
        device=device,
        workers=0,
        name="emotion_yolo12s",

        hsv_h=0.05,
        hsv_s=0.4,
        hsv_v=0.2,
        scale=0.15,
        translate=0.05,
        fliplr=0.25,
        flipud=0.01,

        blur=0.15,
        noise=0.15,
        mosaic=0.02,
        mixup=0.02,


        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,

        val=True,
        save=True,
        save_period=20,
        cache=True  # False, если оперативной памяти < 20GB
    )

if __name__ == "__main__":
    main()
