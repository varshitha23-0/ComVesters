from ultralytics import YOLO
from PIL import Image
import os
import torch
import gc
torch.set_num_threads(1)
# Load model once
model = None

def get_model():
    global model
    if model is None:
        # safer path resolution so it works on Render
        weights_path = os.path.join(os.path.dirname(__file__), "best.pt")
        model = YOLO(weights_path)  # load once, then reuse
        model.fuse()
    return model


# Class-specific information for waste types
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


def detect_image(image_path):
    image = Image.open(image_path).convert("RGB")
    m = get_model()
    results = None 
    try:
        results = m.predict(
            source=image,
            imgsz=640,
            device='cpu',
            verbose=False
        )

        output = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = m.names[class_id]   

                info = class_info.get(class_name, {})

                output.append({
                    "class": class_name,
                    "confidence": round(confidence, 2),
                    "info": info
                })

        return output

    finally:
        del image
        if results is not None:
            del results
       
        gc.collect()
