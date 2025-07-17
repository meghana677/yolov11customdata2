from ultralytics import YOLO

def main():
    # Load your trained model
    model = YOLO("yolov11_custom1.pt")  # Make sure this file exists in your directory

    # Run prediction on an image
    model.predict(
        source="1.mp4",     # Path to your test image
        show=True,          # Show the image in a window
        save=True,          # Save prediction result to 'runs/predict'
                   # Confidence threshold
        line_width=1        # Thinner bounding boxes
    )

if __name__ == "__main__":
    main()
