# Grass Analysis GUI Application

This directory contains the PyQt6 GUI application for the grass coverage and density analysis tool.

## Structure

- `main_gui.py`: Entry point for the GUI application.
- `ui/`: Contains UI related modules.
  - `main_window.py`: Defines the main application window class.
- `core/`: Contains the core logic that connects the UI to the backend analysis scripts.
  - `analysis_runner.py`: Handles running the analysis process in a separate thread.
- `utils/`: Utility modules for the GUI.
  - `logging_handler.py`: Redirects Python logging to the GUI log window.
- `resources/`: Static resources like icons (currently empty).
- `config/`: Configuration files (currently empty).
- `requirements.txt`: Python dependencies for the GUI.
- `README.md`: This file.

## Prerequisites

- Python 3.x
- PyQt6
- Other dependencies listed in `requirements.txt` (potentially numpy, opencv-python, matplotlib, scikit-learn based on the analysis scripts).

## How to Run

1.  **Navigate to the project root directory** (the one containing `app` and `scripts`).
2.  **Install dependencies:**
    ```bash
    pip install -r app/requirements.txt
    # You might also need dependencies from the main project if not already installed
    ```
3.  **Run the GUI application:**
    ```bash
    python app/main_gui.py
    ```

## Features

- Select single image files or a folder containing images.
- Configure analysis parameters:
    - Analysis model (Traditional / Deep Learning)
    - Segmentation method (HSV for Traditional)
    - Calibration option
    - Save debug images
    - Calculate density
    - Output plot layout
- Optional lidar-based height analysis with configurable DBSCAN parameters.
- Select output directory for results.
- Start and monitor the analysis process via a progress bar and log window.
- View analysis summary in the log window upon completion.
- Basic error handling and reporting.
