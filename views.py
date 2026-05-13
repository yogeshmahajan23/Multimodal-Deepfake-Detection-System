from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from datetime import datetime
from DeepByte1.models import Contact
from django.contrib.messages import constants as messages
from django.contrib import messages
from .models import Article  
from newspaper import Article
import os
from DeepByte1.utils import predict_video
import logging
import numpy as np
import cv2
import tempfile
from DeepByte1.utils import preprocess_video 
import time  
import tensorflow as tf
import hashlib
import glob
import pandas as pd
import torchaudio

from safetensors.torch import load_file
from django.shortcuts import render, redirect
from .models import EXEFile
from .forms import EXEFileForm

from django import forms
from huggingface_hub import snapshot_download
import soundfile as sf
import subprocess
import traceback
import soundfile as sf   # if you switched to soundfile
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Download the pretrained deepfake detector model
snapshot_download(
    repo_id="garystafford/wav2vec2-deepfake-voice-detector",
    local_dir="Y:/Django/project1/DeepByte/DeepByte1/ai_models/audioModel",
    ignore_patterns=["*.msgpack"]  # optional, skips large unused files
)


def clean_file(self):
    uploaded_file = self.cleaned_data.get('file')
    if not uploaded_file.name.endswith('.exe'):
        raise forms.ValidationError("Only .exe files are allowed!")
    return uploaded_file



def upload_exe(request):
    if request.method == 'POST':
        print("DEBUG: Upload button clicked!")
        form = EXEFileForm(request.POST, request.FILES)  # ✅ Make sure request.FILES is included!
        if form.is_valid():
            form.save()
            print("DEBUG: File uploaded successfully!")
            return JsonResponse({"success": True})
        else:
            print("DEBUG: Form validation failed!", form.errors)
            return JsonResponse({"success": False, "error": str(form.errors)}, status=400)

    # GET request fallback
    form = EXEFileForm()
    files = EXEFile.objects.all()
    return render(request, "viewBS.html", {"form": form, "files": files})

def list_exe_files(request):
    """Retrieve all uploaded EXE files and display them."""
    files = EXEFile.objects.all()  # Fetch all EXE files from the database
    return render(request, "list_exe_files.html", {"files": files})

MODEL_PATH = "Y:/Django/project/project1/DeepByte/DeepByte1/ai_models/modelv0.2b.tflite"


try:
    interpreter = tf.lite.Interpreter(model_path="Y:/Django/project/project1/DeepByte/DeepByte1/ai_models/modelv0.2b.tflite")
    interpreter.allocate_tensors()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model failed to load: {str(e)}")


UPLOAD_DIR = "Y:/Django/project1/DeepByte/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔹 Generate SHA-256 hash for file content (to uniquely identify each video)
def get_file_hash(file_obj):
    hasher = hashlib.sha256()
    file_obj.seek(0)  # Reset before hashing
    for chunk in file_obj.chunks():
        hasher.update(chunk)
    return hasher.hexdigest()

# ✅ Cache analyzed results to avoid re-processing the same videos
RESULT_CACHE = {}

id2label = {0: "Fake", 1: "Real"}

@csrf_exempt
def analyzeVideo(request):
    """Handles deepfake video detection using TensorFlow Lite model."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method!"}, status=405)

    try:
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            print("❌ ERROR: No file uploaded")
            return JsonResponse({"error": "No file uploaded"}, status=400)

        print(f"✅ File received: {uploaded_file.name}")

        file_hash = get_file_hash(uploaded_file)

        # 🔁 Return cached result if the same video has already been processed
        if file_hash in RESULT_CACHE:
            print(f"🔁 Returning cached result for: {file_hash}")
            return JsonResponse(RESULT_CACHE[file_hash])

        # ✅ Run actual deepfake classification using ML model
        print("🔄 Running deepfake classification...")
        prediction = predict_video(uploaded_file)  # ✅ Calls correct function!

        if "error" in prediction:
            print(f"❌ ERROR in model classification: {prediction['error']}")
            return JsonResponse({"error": prediction["error"]}, status=500)

        # ✅ Apply label mapping to ensure correct classification
        id2label = {0: "Fake", 1: "Real"}
        if "deepfake_classification" in prediction:
            raw_probs = prediction["deepfake_classification"]
            prediction = {id2label[i]: round(raw_probs[i] * 100, 2) for i in range(len(raw_probs))}

        # ✅ Ensure output shape matches model requirements
        expected_shape = (1, 128, 128, 3)
        if isinstance(prediction, np.ndarray) and prediction.shape != expected_shape:
            print(f"❌ Shape mismatch: Expected {expected_shape}, got {prediction.shape}")
            return JsonResponse({"error": f"Model input shape mismatch. Expected {expected_shape}, got {prediction.shape}"}, status=500)

        response_data = {"deepfake_classification": prediction}
        RESULT_CACHE[file_hash] = response_data  # ✅ Cache result

        print("✅ Classification complete.")
        return JsonResponse(response_data, status=200)

    except Exception as e:
        print(f"❌ FATAL ERROR in analyzeVideo: {str(e)}")
        return JsonResponse({"error": f"Internal server error: {str(e)}"}, status=500)
    

@csrf_exempt
def viewBS(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method!"}, status=405)

    # Process binary structure and return response
    return JsonResponse({"binary_data": "Sample binary output"})


@csrf_exempt
def viewBS(request):
    """Handles binary file visualization with improved error handling"""
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            print("DEBUG: No file uploaded!")
            return JsonResponse({"error": "No file uploaded!"}, status=400)

        try:
            # Read binary file safely
            binary_data = uploaded_file.read().hex()[:500]
            print(f"DEBUG: Extracted binary data - {binary_data[:100]}")  # Logs first 100 characters

            return JsonResponse({"binary_data": binary_data}, status=200)
        except Exception as e:
            print(f"DEBUG: Binary processing error - {str(e)}")
            return JsonResponse({"error": f"Binary processing error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method!"}, status=405)


@csrf_exempt
def viewFile(request):
    """Handles binary file visualization with improved error handling"""
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            print("DEBUG: No file uploaded!")
            return JsonResponse({"error": "No file uploaded!"}, status=400)

        try:
            # Read binary file safely
            binary_data = uploaded_file.read().hex()[:500]
            print(f"DEBUG: Extracted binary data - {binary_data[:100]}")  # Logs first 100 characters

            return JsonResponse({"binary_data": binary_data}, status=200)
        except Exception as e:
            print(f"DEBUG: Binary processing error - {str(e)}")
            return JsonResponse({"error": f"Binary processing error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method!"}, status=405)





# Load dataset
DATASET_PATH = "Y:/Django/project/project1/DeepByte/DeepByte1/ai_models/archive/KAGGLE/DATASET-balanced.csv"
df = pd.read_csv(DATASET_PATH)




# ✅ FFmpeg executable path (absolute, bypass PATH issues)
FFMPEG_PATH = r"Y:\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"

# ✅ Model + Upload directories
MODEL_DIR = r"Y:\Django\project1\DeepByte\DeepByte1\ai_models\audioModel"
UPLOAD_DIR = r"Y:\Django\project1\DeepByte\uploads\audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Load feature extractor + model once at startup
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def convert_to_wav(input_file):
    """Convert any audio file (MP3/OGG/Opus) to WAV using FFmpeg."""
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    cmd = [
        FFMPEG_PATH,
        "-y", "-i", input_file,
        "-ar", "16000",  # resample to 16kHz
        "-ac", "1",      # mono
        output_file
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("DEBUG: FFmpeg stdout:", result.stdout.decode(errors="ignore"))
        print("DEBUG: FFmpeg stderr:", result.stderr.decode(errors="ignore"))
        print(f"DEBUG: Converted {input_file} → {output_file}")
        print("DEBUG: WAV exists?", os.path.exists(output_file))
        return output_file
    except subprocess.CalledProcessError as e:
        print("ERROR: FFmpeg conversion failed:", e.stderr.decode())
        raise RuntimeError("FFmpeg conversion failed")


@csrf_exempt
def detect_fake_audio(request):
    """Django view: handle audio upload and return confidence scores + debug info."""
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Only POST method is accepted."}, status=405)

        if "audio" not in request.FILES:
            return JsonResponse({"error": "No audio file provided."}, status=400)

        # ✅ Save uploaded file
        audio_file = request.FILES["audio"]
        filename = os.path.basename(audio_file.name)
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb+") as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # ✅ Convert to WAV before analysis
        wav_path = convert_to_wav(file_path)

        # ✅ Run analysis
        result, debug_info = analyzeAudio(wav_path)
        print("DEBUG: API response =", result)
        print("DEBUG: Debug info =", debug_info)

        return JsonResponse({
            "filename": filename,
            "analysis": result,
            "debug": debug_info
        }, status=200)

    except Exception as e:
        print("ERROR in detect_fake_audio:", str(e))
        print(traceback.format_exc())
        return JsonResponse({
            "error": f"Audio detection failed: {str(e)}",
            "analysis": {},
            "debug": {}
        }, status=500)


def analyzeAudio(audio_path):
    """Analyze audio file and return confidence scores + debug info."""
    try:
        print("DEBUG: Analyzing file:", audio_path)

        # ✅ Load waveform with soundfile (always WAV now)
        waveform, sr = sf.read(audio_path)
        print(f"DEBUG: Loaded waveform shape={waveform.shape}, sample_rate={sr}")

        if waveform.size == 0:
            raise ValueError("Empty waveform loaded — invalid audio file.")

        # ✅ Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)
            print("DEBUG: Converted stereo to mono")

        # ✅ Prepare input
        inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

        # ✅ Run inference
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        # ✅ Map results to labels
        result = {
            model.config.id2label[idx].lower(): round(probs[idx] * 100, 2)
            for idx in range(len(probs))
        }

        # ✅ Collect debug info
        debug_info = {
            "waveform_shape": str(waveform.shape),
            "sampling_rate": sr,
            "logits": logits.tolist(),
            "probs": probs,
            "labels": model.config.id2label
        }

        return result, debug_info

    except Exception as e:
        print("ERROR in analyzeAudio:", str(e))
        print(traceback.format_exc())
        return {"error": f"Audio analysis failed: {str(e)}"}, {
            "waveform_shape": None,
            "sampling_rate": None,
            "logits": None,
            "probs": None,
            "labels": None
        }
    
def search_view(request):
    query = request.GET.get("q", "").strip()
    
    if not query:
        return JsonResponse({"results": []})

    results = Article.objects.filter(title__icontains=query)  
    return JsonResponse({"results": list(results.values_list("title", flat=True))})

def index(request):
    return render(request, "index.html")

def file(request):
    return render(request, "file.html")

def history(request):
    return render(request, "history.html")

def help(request):
    return render(request, "help.html")

def about(request):
    return render(request, "about.html")

def contactus(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        desc = request.POST.get("desc")

        contact = Contact(name=name, email=email, phone=phone, desc=desc, date=datetime.today())
        contact.save()
        
        messages.success(request, "Your message has been sent!")
    
    return render(request, "contactus.html")

