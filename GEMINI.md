# Gemini Context File for Y.O.P.O. Project

This file provides context for the Gemini AI assistant to maintain continuity across sessions.

## Project Overview

**Project Goal:** The primary goal is to build and maintain a web-based inference application for fine-tuned YOLOv8 models designed for UI element detection. The project also includes the training scripts and example usage.

**Key Technologies:**
- **Programming Language:** Python
- **Web Framework:** FastAPI
- **ML Library:** `ultralytics` for YOLOv8
- **Environment:** Conda

## Conda Environment

- **Environment Name:** `yopo`
- **Activation Command:** `conda activate yopo`
- **Important:** Always run Python and pip commands within this environment, for example: `conda run -n yopo pip install ...` or `conda run -n yopo python3 ...`.

## Project Structure

```
/
├── src/                  # Main application directory
│   ├── models/           # Cached models (ignored by git)
│   ├── static/           # Static assets for the web app
│   ├── templates/        # HTML templates
│   ├── main.py           # FastAPI application
│   └── requirements.txt  # App dependencies
├── training/             # Model training assets
│   └── main.ipynb        # Jupyter notebook for training
├── .gitignore            # Git ignore file
├── init.sh               # Script to initialize the conda environment
├── README.md             # Project documentation
├── example.py            # Simple command-line inference script
└── GEMINI.md             # This context file
```

## User Preferences & Workflow

- **Directory Naming:** The user prefers `src` for the main application directory instead of `app`.
- **Model Handling:** Models should be downloaded on-demand during inference and cached locally. They should not be committed to the git repository.
- **Documentation:** The user values clear and comprehensive documentation, including a `README.md` with setup and usage instructions.
- **Examples:** The user appreciates simple example scripts (`example.py`) to demonstrate core functionality.

## Current State (as of last interaction)

- The FastAPI web application is complete and running.
- The `README.md` file has been created and includes a "Future Work" section with a TODO list.
- An `example.py` script for command-line inference has been created.
- The application is running in the background, and the user can access it at `http://127.0.0.1:8000`.
