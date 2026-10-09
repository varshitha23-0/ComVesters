# ComVesters ♻️

An AI-based waste detection system that uses **YOLOv8** to identify waste categories from images and provide basic information and cleanup alerts.

## About the Project

ComVesters explores how computer vision can support waste identification and awareness. Users can upload an image, detect waste items, view their categories, and receive related information and cleanup alerts.

## Features

- **AI-Powered Detection:** Uses YOLOv8 to detect waste in images.
- **Waste Classification:** Identifies categories such as plastic, medical, chemical, and organic waste, depending on the trained model.
- **Waste Information:** Displays basic information about detected waste.
- **Cleanup Alerts:** Provides alerts related to detected waste to support cleanup awareness.
- **Web Interface:** Built with Flask for image submission and displaying detection results.

## Tech Stack

- **Language:** Python
- **Object Detection:** YOLOv8
- **Backend:** Flask
- **Frontend:** HTML, CSS, and Jinja2 templates
- **Image Processing:** OpenCV / Pillow, if used in the implementation

## Project Structure

```text
ComVesters/
├── app/
│   ├── templates/
│   ├── uploads/
│   ├── app.py
│   └── __init__.py
├── README.md
└── requirements.txt
```

*The structure may vary slightly depending on your repository.*

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/varshitha23-0/ComVesters
cd ComVesters
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app/app.py
```

Open the local URL displayed in the terminal in your browser.

## How It Works

1. Upload an image through the web interface.
2. The YOLOv8 model processes the image and detects waste.
3. Detected waste categories and related information are displayed.
4. Cleanup alerts help bring attention to waste that may require action.

## Purpose

The project demonstrates the application of object detection and web development to a practical environmental-awareness problem.

## Future Improvements

- Improve detection accuracy across different waste types.
- Expand the dataset and waste categories.
- Add location-based cleanup reporting.
- Improve deployment performance and handling of uploaded images.

## Disclaimer

Detection results depend on the trained model and image quality. The system is intended as a waste-awareness aid and should not replace professional procedures for handling medical or chemical waste.
