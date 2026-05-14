import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

app = FastAPI(title="Nepali Food Calorie API")


NUTRITION_DB = {
    "chatamari": {"calories": 350, "protein": 10, "carbs": 55, "fat": 10, "note": "Standard serving with meat/egg toppings."},
    "chhoila": {"calories": 275, "protein": 22, "carbs": 7, "fat": 18, "note": "Grilled spicy meat. Excellent lean protein."},
    "dalbhat": {"calories": 600, "protein": 20, "carbs": 95, "fat": 12, "note": "Full thali: Rice + Dal + Tarkari."},
    "dhindo": {"calories": 345, "protein": 11, "carbs": 70, "fat": 3, "note": "Millet/Buckwheat mash. High complex carbs."},
    "gundruk": {"calories": 60, "protein": 6, "carbs": 12, "fat": 1, "note": "Fermented leafy greens. Very low calorie."},
    "kheer": {"calories": 280, "protein": 5, "carbs": 42, "fat": 6, "note": "Rice pudding. Treat as dessert."},
    "momo": {"calories": 300, "protein": 18, "carbs": 38, "fat": 12, "note": "Standard plate (10 pcs). Steamed is healthier."},
    "sekuwa": {"calories": 300, "protein": 22, "carbs": 3, "fat": 22, "note": "Roasted meat skewers. High protein."},
    "selroti": {"calories": 385, "protein": 5, "carbs": 45, "fat": 12, "note": "Traditional rice doughnut (2 pieces)."}
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_nepali_food_v2.keras")

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    CLASS_NAMES = sorted(list(NUTRITION_DB.keys())) # Ensures order matches training
except Exception as e:
    print(f"Loading Error: {e}")
    MODEL = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image")
    
    try:
        img = Image.open(io.BytesIO(await file.read())).convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img, dtype=np.float32), 0)
        
        # Predict
        preds = MODEL.predict(img_array, verbose=0)[0]
        best_idx = np.argmax(preds)
        food_name = CLASS_NAMES[best_idx]
        confidence = float(preds[best_idx])
        
        # Get Nutrition
        info = NUTRITION_DB.get(food_name, {})
        
        return {
            "food": food_name,
            "confidence": f"{confidence*100:.2f}%",
            "nutrition": info,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)