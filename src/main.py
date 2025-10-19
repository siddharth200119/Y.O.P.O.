
import os
import uuid
import requests
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="src/templates")

# Model directory
MODEL_DIR = Path("src/models")
MODEL_DIR.mkdir(exist_ok=True)

# Model URLs
MODEL_URLS = {
    "yopo_v8n.pt": "https://github.com/siddharth200119/Y.O.P.O./releases/download/1.0.0/yopo_v8n.pt",
    "yopo_v8s.pt": "https://github.com/siddharth200119/Y.O.P.O./releases/download/1.0.0/yopo_v8s.pt",
}

def download_model(model_name: str):
    """Downloads the model if it doesn't exist."""
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        print(f"Downloading {model_name}...")
        url = MODEL_URLS.get(model_name)
        if not url:
            raise ValueError(f"Model {model_name} not found in MODEL_URLS.")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {model_name} to {model_path}")
        except requests.exceptions.RequestException as e:
            # Clean up failed download
            if model_path.exists():
                os.remove(model_path)
            raise RuntimeError(f"Failed to download model {model_name}: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(...),
):
    """Handles image upload, runs inference, and returns results."""
    try:
        # Download model if needed
        download_model(model_name)

        # Define paths
        upload_dir = Path("src/static/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        results_dir = Path("src/static/results")
        results_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_extension = Path(file.filename).suffix
        file_name = f"{uuid.uuid4()}{file_extension}"
        upload_path = upload_dir / file_name
        
        with open(upload_path, "wb") as buffer:
            buffer.write(await file.read())

        # Load model and run prediction
        model = YOLO(MODEL_DIR / model_name)
        
        # Predict and save results
        predict_results = model.predict(
            source=str(upload_path),
            save=True,
            project=str(results_dir),
            name=file_name,
            exist_ok=True,
        )

        # Get paths and data
        annotated_image_path = f"/static/results/{file_name}/{Path(file.filename).name}"
        
        # The predict function may save the image with a different name if the original name is repeated
        # We need to find the actual saved image path
        saved_output_dir = results_dir / file_name
        output_files = list(saved_output_dir.glob(f"*{file_extension}"))
        if not output_files:
             output_files = list(saved_output_dir.glob(f"*.jpg"))

        if output_files:
            annotated_image_path = f"/static/results/{file_name}/{output_files[0].name}"


        # Get JSON output
        results_data = []
        for r in predict_results:
            results_data.extend(r.boxes.data.tolist())

        json_output = [
            {
                "box": [d[0], d[1], d[2], d[3]],
                "confidence": d[4],
                "class": int(d[5]),
                "class_name": model.names[int(d[5])],
            }
            for d in results_data
        ]

        return JSONResponse(
            content={
                "annotated_image_url": annotated_image_path,
                "model_output": json_output,
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
