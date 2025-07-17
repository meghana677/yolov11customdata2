import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable synchronous CUDA errors for debugging

from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")

    model.train(
        data="dataset_custom1.yaml",
        imgsz=412,
        batch=2,
        epochs=100,
        workers=0,
        device=0
    )

if __name__ == "__main__":
    main()
