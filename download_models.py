import os
from ultralytics import YOLO

def download_all_models():
    # List of all YOLOv8 model variants you requested
    models_to_download = [
        "yolov8n.pt",  # Nano
        "yolov8s.pt",  # Small
        "yolov8m.pt",  # Medium
        "yolov8l.pt",  # Large
        "yolov8x.pt"   # XLarge
    ]
    
    print("Starting bulk download of YOLOv8 pre-trained models...\n")
    
    for model_name in models_to_download:
        print("-" * 50)
        print(f"Downloading/Verifying: {model_name}")
        print("-" * 50)
        
        try:
            # Instantiating the class forces an automatic download if the file is missing
            model = YOLO(model_name)
            print(f"Successfully downloaded and saved: {model_name}\n")
        except Exception as e:
            print(f"Failed to download {model_name}. Error: {e}\n")

    print("All downloads complete! The .pt files are ready in your directory.")

if __name__ == "__main__":
    download_all_models()