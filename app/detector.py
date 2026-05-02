from flask import Flask, render_template, request
from PIL import Image
import requests
import io
import gc

app = Flask(__name__)

# 👉 YOUR HUGGING FACE SPACE URL (IMPORTANT)
HF_API = "https://varshithazz-comvesters.hf.space/"

# Class info
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
        'recycling': 'Hazardous - Requires special disposal',
        'tips': 'Never dispose in regular trash. Use authorized medical waste facilities.'
    },
    'organic': {
        'toxicity': 'Low',
        'years_to_degrade': '30-180 days',
        'environmental_impact': 'Low - Biodegradable',
        'recycling': 'Compostable',
        'tips': 'Compost in bins or bury in soil.'
    },
    'chemical': {
        'toxicity': 'High',
        'years_to_degrade': 'Permanent',
        'environmental_impact': 'Severe - Soil & water contamination',
        'recycling': 'Hazardous waste',
        'tips': 'Never pour into drains. Use safe disposal methods.'
    }
}


# ========================
# 🔥 CORE FUNCTION
# ========================
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
            class_name = item["class"].lower()
            confidence = item["confidence"]

            info = class_info.get(class_name, {})

            output.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "info": info
            })

        return output

    except Exception as e:
        print("API Error:", e)
        return [{"class": "Error", "confidence": 0, "info": {}}]


# ========================
# 🌐 ROUTE
# ========================
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

            results = detect_image(image)

        except Exception as e:
            return render_template("index.html", results=[], error=str(e))

        finally:
            if image:
                del image
            gc.collect()

    return render_template("index.html", results=results)


# ========================
# 🚀 RUN
# ========================
if __name__ == "__main__":
    app.run()
