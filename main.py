import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, File, UploadFile
from pydantic import BaseModel
from ml_core import CustomSignLanguageNN

app = FastAPI()

# Модульная конфигурация системы (Feature flags)
CONFIG = {
    "ml_enabled": True,
    "allow_custom_gestures": True,
    "default_image_size": 64 * 64 * 3  # Сжатое RGB изображение
}

# Инициализация ML
model = CustomSignLanguageNN(input_size=CONFIG["default_image_size"])

# Заглушка базы пользователей
FAKE_DB = {"admin": "secret_token_123"}

def verify_token(token: str):
    if token not in FAKE_DB.values():
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return True

@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if not CONFIG["ml_enabled"]:
                await websocket.send_json({"error": "ML модуль временно отключен"})
                continue
            
            # Извлечение параметров
            fast_mode = data.get("fast_mode", False)
            threshold = data.get("threshold", 0.5)
            use_custom = data.get("use_custom", False)
            
            # Эмуляция предобработки (в реальности тут конвертация Base64 -> NumPy Array)
            # В "быстром режиме" можно использовать уменьшенное разрешение
            dummy_pixels = np.random.rand(1, CONFIG["default_image_size"]) 
            
            prediction, confidence = model.predict(dummy_pixels)
            
            if confidence[0] < threshold:
                result = "Не распознано"
            else:
                result = f"Жест_{prediction[0]}" # Маппинг на буквы/цифры
            
            await websocket.send_json({"gesture": result, "confidence": float(confidence[0])})
            
    except WebSocketDisconnect:
        print("Клиент отключился")

@app.post("/upload/image")
async def upload_image(fast_mode: bool = False, threshold: float = 0.5, file: UploadFile = File(...)):
    # Логика обработки одиночного фото
    return {"status": "ok", "gesture": "А", "confidence": 0.89}

@app.post("/gestures/add")
async def add_custom_gesture(gesture_name: str, authorized: bool = Depends(verify_token)):
    if not CONFIG["allow_custom_gestures"]:
        raise HTTPException(status_code=400, detail="Добавление отключено в конфигурации")
    # Логика добавления в датасет и дообучения
    return {"message": f"Жест '{gesture_name}' успешно добавлен"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)