from ultralytics import YOLO
import torch

def main():
    MODEL_PATH = "models/yolo12s.pt"

    model = YOLO(MODEL_PATH)

    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data="Emotion-2/data.yaml",

        epochs=10,
        imgsz=640,
        batch=25,
        device=device,
        workers=0,
        name="emotion_yolo12s",

        hsv_h=0.05,         # цветовой тон
        hsv_s=0.7,          # насыщенность
        hsv_v=0.4,          # яркость
        flipud=0.05,        # почти не трогаем вертикально
        fliplr=0.5,         # зеркалка
        degrees=10,         # небольшие повороты лица
        translate=0.1,      # небольшие смещения
        scale=0.4,          # масштабирование лица
        shear=0.0,          # не трогаем
        perspective=0.0,    # не трогаем
        mosaic=0.05,        # мягкая мозаика
        mixup=0.05,         # мягкий mixup

        optimizer="SGD",
        lr0=0.002,
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
