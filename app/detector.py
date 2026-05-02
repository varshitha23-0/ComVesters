import requests
import io

# 👉 Replace with your actual Hugging Face Space URL
HF_API = "https://your-space-name.hf.space/predict"

# Class info (keep this)
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
        'environmental_impact': 'Low - Biodegradable, returns to soil',
        'recycling': 'Compostable',
        'tips': 'Compost in bins or bury in soil. Great for gardens!'
    },
    'chemical': {
        'toxicity': 'High to Critical',
        'years_to_degrade': 'Permanent (bioaccumulative)',
        'environmental_impact': 'Severe - Contaminates soil and water',
        'recycling': 'Depends on type - Hazardous waste',
        'tips': 'Store safely, never pour down drains. Check local hazmat disposal.'
    }
}


def detect_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    try:
        response = requests.post(
            HF_API,
            files={"file": buffer},
            timeout=15
        )

        data = response.json()["results"]

        output = []

        for item in data:
            class_name = item["class"]
            confidence = item["confidence"]

            info = class_info.get(class_name, {})

            output.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "info": info
            })

        return output

    except Exception:
        return [{"class": "Error", "confidence": 0, "info": {}}]
