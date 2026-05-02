from flask import Flask, render_template, request
from PIL import Image
import torch
import gc
import io

from app.detector import detect_image, class_info

app = Flask(__name__)

torch.set_num_threads(1)


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

def get_whi_risk_level(whi):
    if whi < 2:
        return "Low (🟢)"
    elif whi < 5:
        return "Moderate (🟡)"
    elif whi < 8:
        return "High (🟠)"
    else:
        return "Critical (🔴)"

def get_cleanup_alert(whi, waste_type):
    if whi >= 8:
        return f"⚠️ Critical {waste_type} detected!"
    elif whi >= 5:
        return f"⚡ High-risk {waste_type} detected!"
    elif whi >= 2:
        return f"📢 Moderate {waste_type} detected."
    else:
        return f"✓ Low-risk {waste_type}."



@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return render_template("index.html", results=[], error="No image uploaded")

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")

            detection_results = detect_image(image)

            if detection_results:
                for d in detection_results:
                    class_name = d["class"]
                    confidence = d["confidence"]
                    info = d["info"]

                    whi = calculate_whi(class_name, confidence)

                    results.append({
                        "class": class_name.capitalize(),
                        "confidence": round(confidence * 100, 2),
                        "whi": f"{whi} - {get_whi_risk_level(whi)}",
                        "toxicity": info.get("toxicity", ""),
                        "years_to_degrade": info.get("years_to_degrade", ""),
                        "environmental_impact": info.get("environmental_impact", ""),
                        "recycling": info.get("recycling", ""),
                        "tips": info.get("tips", ""),
                        "cleanup_alert": get_cleanup_alert(whi, class_name.capitalize())
                    })
            else:
                results = [{"class": "No Objects", "confidence": 0}]

        finally:
            del file
            del image
            gc.collect()

    return render_template("index.html", results=results)


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run()
