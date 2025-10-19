import os
import requests
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
MODEL_NAME = "yopo_v8n.pt"
MODEL_URL = "https://github.com/siddharth200119/Y.O.P.O./releases/download/1.0.0/yopo_v8n.pt"
MODEL_DIR = Path("src/models")
SAMPLE_IMAGE = Path("training/dataset/images/test/creative_www.figma.com_1729631861.png")

def download_model(model_path: Path, url: str):
    """Downloads the model if it doesn't exist."""
    if not model_path.exists():
        print(f"Downloading {model_path.name}...")
        MODEL_DIR.mkdir(exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {model_path.name} to {model_path}")
        except requests.exceptions.RequestException as e:
            if model_path.exists():
                os.remove(model_path)
            raise RuntimeError(f"Failed to download model: {e}")

def run_inference():
    """Runs a simple inference example."""
    print("--- YOLO Inference Example ---")

    # Ensure sample image exists
    if not SAMPLE_IMAGE.exists():
        print(f"Error: Sample image not found at {SAMPLE_IMAGE}")
        print("Please make sure the dataset is available.")
        return

    # Download the model
    model_path = MODEL_DIR / MODEL_NAME
    try:
        download_model(model_path, MODEL_URL)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    # Load the model
    print(f"\nLoading model: {model_path.name}")
    model = YOLO(model_path)

    # Run prediction
    print(f"Running inference on: {SAMPLE_IMAGE.name}")
    results = model.predict(source=str(SAMPLE_IMAGE))

    # Process and print results
    print("\n--- Detection Results ---")
    result = results[0]  # Get results for the first image
    if not result.boxes:
        print("No objects detected.")
        return

    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        coords = box.xyxy.tolist()[0]
        
        print(
            f"  - Class: {class_name} (ID: {class_id})\n"
            f"    Confidence: {confidence:.4f}\n"
            f"    Coordinates (xyxy): [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f}]\n"
        )

if __name__ == "__main__":
    run_inference()
