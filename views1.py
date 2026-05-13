from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification






# Load DeepFake detection model
model_name = "prithivMLmods/deepfake-detector-model-v1"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {0: "fake", 1: "real"}  


@csrf_exempt  #  Allows POST requests without CSRF protection (remove in production)
def analyzeDF(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method!"}, status=405)

    # Process the uploaded file and return a response
    return JsonResponse({"verdict": "Fake", "confidence": 95})


def classify_image(image):

    try:
        image = Image.open(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        prediction = {id2label[i]: round(probs[i] * 100, 2) for i in range(len(probs))}  # Convert to percentage
        return prediction

    except Exception as e:
        print(f"DEBUG: DeepFake classification error - {str(e)}")
        return {"error": f"DeepFake classification error: {str(e)}"}


@csrf_exempt
def analyzeDF(request):
    """Handles DeepFake detection"""
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return JsonResponse({"error": "No file uploaded!"}, status=400)

        try:
            prediction = classify_image(uploaded_file)
            return JsonResponse({"deepfake_classification": prediction}, status=200)
        except Exception as e:
            return JsonResponse({"error": f"DeepFake analysis error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method!"}, status=405)
