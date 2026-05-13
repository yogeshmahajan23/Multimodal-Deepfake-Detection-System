import numpy as np
import cv2
import tensorflow.lite as tflite
import os
import time  
import hashlib
import tempfile
import gc  # For garbage collection


def get_file_hash(file_obj_or_path):
    """Returns SHA-256 hash for a file object or file path."""
    hasher = hashlib.sha256()

    if isinstance(file_obj_or_path, str):  # Handle file paths
        with open(file_obj_or_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    else:  # Handle uploaded file objects
        file_obj_or_path.seek(0)
        for chunk in file_obj_or_path.chunks():
            hasher.update(chunk)

    return hasher.hexdigest()

# Define processed cache directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CACHE_DIR = os.path.join(BASE_DIR, "processed_cache")
os.makedirs(PROCESSED_CACHE_DIR, exist_ok=True)


def preprocess_video(file_path, file_hash, num_frames=10):
    """Extracts `num_frames` evenly spaced frames resized to (128, 128, 3)."""
    try:
        print(f"DEBUG: Loading video → {file_path}")
        processed_file_path = os.path.join(PROCESSED_CACHE_DIR, f"{file_hash}_frames.npy")

        if os.path.exists(processed_file_path):
            frames = np.load(processed_file_path)
            if frames.shape[1:] == (128, 128, 3):
                print(f"📦 Using cached preprocessed frames → {processed_file_path}")
                return frames
            else:
                print(f"⚠️ Cached frame shape mismatch: {frames.shape}, regenerating...")
                os.remove(processed_file_path)

        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        frames = []
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)
                frames.append(frame_resized)
            frame_id += 1

        cap.release()

        if not frames:
            raise ValueError("❌ No valid frames extracted.")

        video_data = np.array(frames, dtype=np.float32)
        np.save(processed_file_path, video_data)
        print(f"✅ Preprocessed frames saved → {processed_file_path}")
        return video_data

    except Exception as e:
        print(f"❌ ERROR in preprocess_video(): {e}")
        return None
    



def predict_video(video_file):
    """Runs deepfake classification using TensorFlow Lite on multiple frames."""
    try:
        print("🔍 Starting video classification...")

        temp_dir = tempfile.gettempdir()
        file_hash = get_file_hash(video_file)
        temp_path = os.path.join(temp_dir, f"{file_hash}.mp4")

        with open(temp_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        frames = preprocess_video(temp_path, file_hash, num_frames=10)
        if frames is None:
            return {"error": "Video preprocessing failed"}
        if len(frames.shape) != 4 or frames.shape[1:] != (128, 128, 3):
            return {"error": f"Preprocessed shape mismatch: {frames.shape}"}

        # Load a fresh interpreter
        model_path = os.path.join(BASE_DIR, "ai_models", "modelv0.2b.tflite")
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        fake_scores, real_scores = [], []

        for frame in frames:
            input_tensor = np.expand_dims(frame, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_tensor.copy())
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index']).copy()
            fake_scores.append(float(output[0][0]))
            real_scores.append(float(output[0][1]))

        del interpreter
        gc.collect()

        fake_avg = round(np.mean(fake_scores) * 100, 2)
        real_avg = round(np.mean(real_scores) * 100, 2)

        print(f"📈 Classification — Fake: {fake_avg:.2f}%, Real: {real_avg:.2f}%")
        return {"fake": fake_avg, "real": real_avg}

    except Exception as e:
        print(f"❌ ERROR in predict_video(): {e}")
        return {"error": f"DeepFake classification error: {str(e)}"}