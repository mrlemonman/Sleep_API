# sleep_tracker_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow_hub as hub
import librosa
import tensorflow as tf
import os
import tempfile
import soundfile as sf
import joblib
import base64
from typing import Dict, List
import shutil

app = FastAPI(title="Sleep Tracker API", version="1.0")

# CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
yamnet_model = None
sneeze_sniff_clf = None
class_names = []
sneeze_sniff_classes = ["sneeze", "sniff", "neither"]

mapping = {
    "snore": ["Snoring"],
    "cough": ["Cough"],
    "fart": ["Fart"],
    "sleep_talking": ["Speech", "Babbling", "Whimper"],
    "sneeze_sniff": [],
    "laughter": ["Laughter"],
    "music": ["Music"]
}

class_thresholds = {
    "snore": 0.3,
    "cough": 0.3,
    "fart": 0.3,
    "sleep_talking": 0.5,
    "sneeze_sniff": 0.7,
    "laughter": 0.1,
    "music": 0.3
}

@app.on_event("startup")
async def load_models():
    global yamnet_model, sneeze_sniff_clf, class_names
    
    print("Loading YAMNet model...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    print("Loading custom classifier...")
    sneeze_sniff_clf = joblib.load("sneeze_sniff_classifier.pkl")
    
    print("Loading class map...")
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    with open(class_map_path, 'r') as f:
        class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]
    
    print("Models loaded successfully!")

def predict_sneeze_sniff(chunk, threshold=0.7):
    _, embeddings, _ = yamnet_model(chunk)
    mean_embedding = np.mean(embeddings.numpy(), axis=0)
    probs = sneeze_sniff_clf.predict_proba([mean_embedding])[0]
    pred_idx = np.argmax(probs)
    pred_prob = probs[pred_idx]
    if pred_prob < threshold or sneeze_sniff_classes[pred_idx] == "neither":
        return None, 0.0
    return "sneeze_sniff", pred_prob

def detect_events(waveform, sr=16000, window_sec=1.0, padding_sec=1.0, top_n=3):
    window_samples = int(window_sec * sr)
    num_windows = len(waveform) // window_samples
    events = {evt: [] for evt in mapping.keys()}
    current_event, current_start, current_chunk = None, 0, []
    current_scores = []
    padding_samples = int(padding_sec * sr)

    for i in range(num_windows):
        start, end = i * window_samples, (i + 1) * window_samples
        chunk = waveform[start:end]
        if len(chunk) == 0:
            continue
        if sr != 16000:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)

        scores, embeddings, _ = yamnet_model(chunk)
        mean_scores = np.mean(scores.numpy(), axis=0)

        silence_idx = class_names.index("Silence")
        if mean_scores[silence_idx] >= 0.5:
            pred_event = None
            confidence = 0.0
        else:
            pred_event, confidence = predict_sneeze_sniff(chunk, threshold=class_thresholds["sneeze_sniff"])
            if not pred_event:
                candidates = {}
                for evt, yam_classes in mapping.items():
                    if evt == "sneeze_sniff":
                        continue
                    idxs = [class_names.index(c) for c in yam_classes if c in class_names]
                    if not idxs:
                        continue
                    score = mean_scores[idxs].max()
                    if score >= class_thresholds[evt]:
                        candidates[evt] = score
                if candidates:
                    pred_event = max(candidates, key=candidates.get)
                    confidence = candidates[pred_event]
                else:
                    confidence = 0.0

        if pred_event == current_event:
            current_chunk.append(chunk)
            current_scores.append(confidence)
        else:
            if current_event is not None:
                clip_audio = np.concatenate(current_chunk)
                timestamp = current_start / sr
                avg_confidence = np.mean(current_scores) if current_scores else 0.0
                events[current_event].append((timestamp, clip_audio, current_start, 
                                             current_start + len(clip_audio), avg_confidence))
            current_event, current_start, current_chunk = pred_event, start, [chunk]
            current_scores = [confidence] if confidence > 0 else []

    if current_event and current_chunk:
        clip_audio = np.concatenate(current_chunk)
        timestamp = current_start / sr
        avg_confidence = np.mean(current_scores) if current_scores else 0.0
        events[current_event].append((timestamp, clip_audio, current_start, 
                                     current_start + len(clip_audio), avg_confidence))

    # Filter top N
    top_events = {}
    for evt, clips in events.items():
        if not clips:
            top_events[evt] = []
            continue
        sorted_clips = sorted(clips, key=lambda x: x[4], reverse=True)
        top_events[evt] = sorted_clips[:top_n]

    # Convert to base64
    result = {}
    for evt, clips in top_events.items():
        result[evt] = []
        for item in clips:
            if isinstance(item, tuple) and len(item) == 5:
                ts, audio, start_sample, end_sample, confidence = item
                
                padded_start = max(0, start_sample - padding_samples)
                padded_end = min(len(waveform), end_sample + padding_samples)
                padded_audio = waveform[padded_start:padded_end]
                
                # Convert to bytes
                temp_path = tempfile.mktemp(suffix='.wav')
                sf.write(temp_path, padded_audio, sr)
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()
                os.remove(temp_path)
                
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                result[evt].append({
                    "timestamp": float(ts),
                    "confidence": float(confidence),
                    "audio_base64": audio_base64
                })

    return result

@app.get("/")
async def root():
    return {
        "message": "Sleep Tracker API",
        "status": "running",
        "version": "1.0",
        "endpoints": {
            "/analyze": "POST - Upload audio file for analysis"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": yamnet_model is not None}

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    top_n: int = 3,
    window_sec: float = 1.0,
    padding_sec: float = 1.0
):
    """
    Analyze sleep audio and return top N events per category.
    
    Parameters:
    - file: Audio file (WAV, MP3, etc.)
    - top_n: Number of top events to return per category (default: 3)
    - window_sec: Analysis window size in seconds (default: 1.0)
    - padding_sec: Padding around events in seconds (default: 1.0)
    
    Returns JSON with detected events, timestamps, confidence scores, and audio clips
    """
    
    if yamnet_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Save uploaded file
    temp_input = tempfile.mktemp(suffix=os.path.splitext(file.filename)[1])
    try:
        with open(temp_input, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Load audio
        waveform, sr = librosa.load(temp_input, sr=None, mono=True)
        
        # Detect events
        results = detect_events(waveform, sr=sr, window_sec=window_sec, 
                               padding_sec=padding_sec, top_n=top_n)
        
        # Count total events
        total_events = sum(len(events) for events in results.values())
        
        return {
            "status": "success",
            "total_events": total_events,
            "duration_seconds": len(waveform) / sr,
            "events": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
