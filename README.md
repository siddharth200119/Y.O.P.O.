# Y.O.P.O. - You Only Predict Once

This project contains fine-tuned YOLOv8 models for UI element detection, along with a web-based inference application and training scripts.

## Getting Started

### 1. Initial Setup

First, you need to create the Conda environment and install Jupyter. The `init.sh` script automates this process.

```bash
bash init.sh
```

This will:
- Create a Conda environment named `yopo`.
- Install Jupyter Lab and Notebook.

### 2. Install Application Dependencies

Next, install the necessary Python packages for the inference web application.

```bash
# Activate the conda environment
conda activate yopo

# Install dependencies
pip install -r src/requirements.txt
```

## Running the Inference Web App

The web application allows you to upload an image and get object detection results from the fine-tuned YOLO models.

1.  **Activate the Conda environment:**
    ```bash
    conda activate yopo
    ```

2.  **Run the web server:**
    ```bash
    python3 src/main.py
    ```

3.  **Access the application:**
    Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).

The first time you select a model for inference, it will be downloaded and cached in the `src/models` directory.

## Running a Simple Inference

An `example.py` script is provided for a quick command-line inference test.

1.  **Activate the Conda environment:**
    ```bash
    conda activate yopo
    ```

2.  **Run the example:**
    ```bash
    python3 example.py
    ```

This will download the YOLOv8n model (if not already present), run inference on a sample image, and print the detection results to the console.

## Training

The models in this project were trained using the `training/main.ipynb` notebook. If you want to retrain the models or train with new data, you can use this notebook as a starting point.

## Future Work

- [ ] **Custom Dataset Collection Script:** Develop a script to automate the collection and annotation of custom datasets for training.
- [ ] **Hierarchical Detection Model:** Implement a secondary model that takes a detected bounding box (e.g., a button or a card) and detects the text and icons within it.
- [ ] **Stable Data Output:** Investigate and implement models or techniques that provide more stable and structured data output, reducing variability between inferences.
- [ ] **Python Package:** Package the project into an installable Python package (`pip install yopo`) for easier distribution and use.