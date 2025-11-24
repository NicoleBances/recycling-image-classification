#Este es la API, se ejecuta aparte, carga el modelo y recibe peticiones del robot NAO

import uvicorn
from fastapi import FastAPI, Request, Header
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2 
import io

app = FastAPI()

print("Cargando modelo...")
try:
    model = load_model('best_model_2.h5')
    print("Modelo cargado.")
except:
    model = None

IMG_HEIGHT = 224
IMG_WIDTH = 224

@app.post("/predict_lite/")
async def predict_lite(
    request: Request, 
    x_width: int = Header(..., alias="X-Image-Width"), 
    x_height: int = Header(..., alias="X-Image-Height")
):
    if model is None:
        return {"success": False, "error": "Modelo no cargado"}

    try:
        raw_bytes = await request.body()
        img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_original = img_array.reshape((x_height, x_width, 3))
        
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        cv2.imwrite("ultima_vision_nao.jpg", img_bgr)
        print("FOTO GUARDADA: Revisa el archivo 'ultima_vision_nao.jpg' en tu carpeta.")
        
        # Continuar con el proceso normal...
        image_pil = Image.fromarray(img_original)
        image_resized = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        
        input_data = np.array(image_resized) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        prediction = model.predict(input_data)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        labels = ["biodegradable", "vidrio", "metal", "no reciclable", "papel", "plástico"] # ¡Recuerda ajustar esto!
        result_text = labels[class_index] if class_index < len(labels) else "Desconocido"

        return {
            "success": True,
            "label": result_text,
            "confidence": confidence
        }

    except Exception as e:
        print("Error:", str(e))
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)