from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import uuid
import os
import numpy as np
import librosa

from transformers import pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔥 LAZY LOADING (RAM FIX)
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = pipeline(
            "audio-classification",
            model="superb/hubert-large-superb-er"
        )
    return classifier


class AudioRequest(BaseModel):
    audio_base64: str


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/detect-voice")
def detect_voice(data: AudioRequest):
    if not data.audio_base64:
        raise HTTPException(status_code=400, detail="Audio missing")

    try:
        # 1️⃣ Base64 → audio file
        audio_bytes = base64.b64decode(data.audio_base64)
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        # 2️⃣ Load audio
        audio, sr = librosa.load(filepath, sr=16000)

        # 🔥 Force minimum audio length (1 sec padding)
        min_length = sr
        if len(audio) < min_length:
            pad_width = min_length - len(audio)
            audio = np.pad(audio, (0, pad_width))

        # 3️⃣ Load model ONLY when needed
        clf = get_classifier()
        result = clf(audio)[0]

        raw_label = result["label"].lower()
        confidence = float(result["score"])

        # 4️⃣ Simple mapping
        if "synthetic" in raw_label or "spoof" in raw_label:
            prediction = "AI_GENERATED"
        else:
            prediction = "HUMAN"

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "message": "Voice analysis completed"
    }
