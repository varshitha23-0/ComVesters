from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from detector import detect_image, class_info
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model from parent directory
# Path: ../best.pt means go one folder up from app/ to find best.pt

# WHI (Waste Hazard Index) Calculation Weights
TOXICITY_WEIGHTS = {
    'medical': 8.0,
    'chemical': 8.5,
    'plastic': 9.0,
    'organic': 1.5
}

# Environmental factors (base values, can be modified based on location/weather)
ENVIRONMENTAL_FACTORS = {
    'proximity_to_water': 2.0,   # near water bodies = higher risk
    'population_density': 1.8,   # near residential areas = higher risk
    'weather_condition': 1.5,    # rain/wind = higher dispersal risk
    'temperature': 1.2           # affects degradation speed
}

def calculate_whi(waste_type, confidence, quantity_factor=1.0, environmental_score=1.0):
    """
    Calculate Waste Hazard Index (WHI)
    
    Parameters:
    - waste_type: type of waste detected (plastic, medical, organic, chemical)
    - confidence: detection confidence (0-1 scale) - represents concentration
    - quantity_factor: multiplier for detected quantity (default 1.0, can be 0.5-3.0)
    - environmental_score: multiplier for environmental risk (default 1.0, range 0.5-2.5)
    
    WHI Calculation Formula:
    WHI = (Toxicity Weight x Confidence x Quantity Factor x Environmental Score)
    
    WHI Scale:
    - 0-2: Low Risk (Green)
    - 2-5: Moderate Risk (Yellow)
    - 5-8: High Risk (Orange)
    - 8-10: Critical Risk (Red)
    """
    
    # Get toxicity weight for waste type
    toxicity = TOXICITY_WEIGHTS.get(waste_type.lower(), 3.0)
    
    # Base WHI calculation
    whi = toxicity * confidence * quantity_factor * environmental_score
    
    # Cap WHI at 10.0
    whi = min(max(whi, 0), 10.0)
    
    return round(whi, 2)

def get_whi_risk_level(whi):
    """Determine risk level based on WHI score"""
    if whi < 2:
        return "Low (🟢)"
    elif whi < 5:
        return "Moderate (🟡)"
    elif whi < 8:
        return "High (🟠)"
    else:
        return "Critical (🔴)"

def get_cleanup_alert(whi, waste_type):
    """Generate cleanup alert based on WHI and waste type"""
    if whi >= 8:
        return f"⚠️ URGENT: Critical {waste_type} waste detected. Immediate cleanup required!"
    elif whi >= 5:
        return f"⚡ WARNING: High-risk {waste_type} waste detected. Schedule cleanup soon."
    elif whi >= 2:
        return f"📢 Moderate {waste_type} waste detected. Plan cleanup within a few days."
    else:
        return f"✓ Low-risk {waste_type} waste. Standard disposal procedures."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    image_url = None

    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", results=[], error="No file selected")
        
        file = request.files["image"]

        if file and file.filename != "" and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            image_url = f"/uploads/{filename}"

            try:
                # Run YOLOv8 detection
                detection_results = detect_image(path)
                
                if detection_results:
                    for detection in detection_results:
                        class_name = detection['class']
                        confidence = detection['confidence']
                        info = detection['info']

                        # Calculate WHI (using confidence as quantity indicator)
                        whi = calculate_whi(
                            waste_type=class_name,
                            confidence=confidence,
                            quantity_factor=1.0, 
                            environmental_score=1.0  
                        )
                        whi_risk = get_whi_risk_level(whi)
                        cleanup_alert = get_cleanup_alert(whi, class_name.capitalize())
                            
                        results.append({
                            'class': class_name.capitalize(),
                            'confidence': round(confidence * 100, 2),
                            'whi': f"{whi} - {whi_risk}",
                            'toxicity': info['toxicity'],
                            'years_to_degrade': info['years_to_degrade'],
                            'environmental_impact': info['environmental_impact'],
                            'recycling': info['recycling'],
                            'tips': info['tips'],
                            'cleanup_alert': cleanup_alert
                        })
                else:
                    results = [{'class': 'No Objects', 'confidence': 0, 'message': 'No waste detected in image'}]
            except Exception as e:
                return render_template("index.html", results=[], error=f"Detection error: {str(e)}", image_url=image_url)

    return render_template("index.html", results=results, image_url=image_url, error=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)