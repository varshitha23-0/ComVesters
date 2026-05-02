from flask import Flask, render_template, request
from PIL import Image
import requests
import io
import gc

app = Flask(__name__)


HF_API = "https://varshithazz-comvesters.hf.space/predict"

# Class metadata
class_info = {
    'plastic': {
        'toxicity': 'High',
        'years_to_degrade': '450+ years',
        'environmental_impact': 'Severe - Pollutes oceans, harms wildlife',
        'recycling': 'Recyclable in most areas',
        'tips': 'Reduce plastic use, reuse containers, choose biodegradable alternatives'
    },
    'medical': {
        'toxicity': 'Critical',
        'years_to_degrade': '100+ years',
        'environmental_impact': 'Critical - Biohazard, toxic chemicals',
        'recycling': 'Hazardous waste disposal required',
        'tips': 'Do not mix with regular waste. Use medical disposal bins.'
    },
    'organic': {
        'toxicity': 'Low',
        'years_to_degrade': '30-180 days',
        'environmental_impact': 'Low - Biodegradable',
        'recycling': 'Compostable',
        'tips': 'Use for composting'
    },
    'chemical': {
        'toxicity': 'High',
        'years_to_degrade': 'Permanent',
        'environmental_impact': 'Severe contamination risk',
        'recycling': 'Hazardous waste only',
        'tips': 'Handle with care and dispose via authorized centers'
    }
}

# ---------------------------
# WHI CALCULATION
# ---------------------------
TOXICITY_WEIGHTS = {
    'plastic': 9.0,
    'medical': 8.0,
    'chemical': 8.5,
    'organic': 1.5
}

def calculate_whi(waste_type, confidence):
    toxicity = TOXICITY_WEIGHTS.get(waste_type.lower(), 3.0)
    whi = toxicity * confidence
    return round(min(max(whi, 0), 10), 2)

def get_risk(whi):
    if whi < 2:
        return "Low (🟢)"
    elif whi < 5:
        return "Moderate (🟡)"
    elif whi < 8:
        return "High (🟠)"
    else:
        return "Critical (🔴)"

def get_alert(whi, waste_type):
    if whi >= 8:
        return f"⚠️ Critical {waste_type} detected!"
    elif whi >= 5:
        return f"⚡ High-risk {waste_type} detected!"
    elif whi >= 2:
        return f"📢 Moderate {waste_type} detected."
    else:
        return f"✓ Low-risk {waste_type}."


# ---------------------------
# HF API CALL (YOLO RESULT)
# ---------------------------
def detect_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    try:
        response = requests.post(
            HF_API,
            files={"file": buffer},
            timeout=20
        )

        data = response.json().get("results", [])

        output = []
        for item in data:
            class_name = item["class"]
            confidence = item["confidence"]

            info = class_info.get(class_name.lower(), {})

            output.append({
                "class": class_name,
                "confidence": confidence,
                "info": info
            })

        return output

    except Exception as e:
        return [{"class": "Error", "confidence": 0, "info": {"tips": str(e)}}]


# ---------------------------
# ROUTE
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return render_template("index.html", results=[], error="No image uploaded")

        image = None

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")

            detections = detect_image(image)

            for d in detections:
                class_name = d["class"]
                confidence = float(d["confidence"])
                info = d["info"]

                whi = calculate_whi(class_name, confidence)

                results.append({
                    "class": class_name.capitalize(),
                    "confidence": round(confidence * 100, 2),
                    "whi": f"{whi} - {get_risk(whi)}",
                    "toxicity": info.get("toxicity", ""),
                    "years_to_degrade": info.get("years_to_degrade", ""),
                    "environmental_impact": info.get("environmental_impact", ""),
                    "recycling": info.get("recycling", ""),
                    "tips": info.get("tips", ""),
                    "cleanup_alert": get_alert(whi, class_name.capitalize())
                })

        except Exception as e:
            return render_template("index.html", results=[], error=str(e))

        finally:
            if image:
                del image
            gc.collect()

    return render_template("index.html", results=results)


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
