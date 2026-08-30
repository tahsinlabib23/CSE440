from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import unicodedata
import re
import os

app = FastAPI()



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None

class PredictionRequest(BaseModel):
    case_title: str
    case_text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = None

def clean_legal_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.get("/api/health")
def health_check():
    if model is None or vectorizer is None:
        return {"status": "error", "message": "Models failed to load."}
    return {"status": "ok", "message": "API is running and models are loaded."}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Preprocess input exactly as done during training
        model_text = "TITLE: " + request.case_title + " TEXT: " + request.case_text
        cleaned_text = clean_legal_text(model_text)
        
        # If the input is completely empty after cleaning
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Input text is empty after cleaning.")

        # Vectorize
        features = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Optional: get confidence/probabilities if available
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
            
        return PredictionResponse(prediction=prediction, confidence=confidence)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
