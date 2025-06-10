import torch
import timm
import cv2
import numpy as np
import tempfile
import os
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import PIL.Image as Image

app = Flask(__name__)

# Load the pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("tf_efficientnet_b4", pretrained=True, num_classes=1)
model.eval().to(device)

# Detection Thresholds
THRESHOLD_IMAGE = 0.7
THRESHOLD_VIDEO = 0.6
FAKE_VIDEO_PERCENTAGE = 40  # Percentage of frames classified as fake to label a video as deepfake

# Image preprocessing function
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0).to(device)

# Predict function for a single frame
def predict_frame(frame):
    img_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model(img_tensor)
    return torch.sigmoid(output).item()

# Process video efficiently
def process_video(video_bytes):
    try:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_bytes)
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            os.remove(temp_video.name)
            return "Error: Empty or corrupt video file!"

        # Sample frames randomly (10-15 frames)
        frame_count = min(15, total_frames)
        frame_indices = np.random.choice(range(total_frames), size=frame_count, replace=False)
        fake_count = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            prediction = predict_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(f"Frame {idx}: Prediction = {prediction:.2f}")  # Debugging
            if prediction > THRESHOLD_VIDEO:
                fake_count += 1

        cap.release()
        os.remove(temp_video.name)  # Delete temp file after processing
        
        fake_percentage = (fake_count / frame_count) * 100
        return "Fake Video (Deepfake)" if fake_percentage > FAKE_VIDEO_PERCENTAGE else "Real Video"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Detect fake image or video
def detect_fake(file):
    try:
        file_bytes = file.read()

        # Check if it's an image
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            img = np.array(img)
            prediction = predict_frame(img)
            return "Fake Image (Deepfake)" if prediction > THRESHOLD_IMAGE else "Real Image"
        except Exception:
            pass  # Not an image, check for video

        # Process video
        return process_video(file_bytes)
    except Exception as e:
        return f"Error: {str(e)}"

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded!"})

            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected!"})

            result = detect_fake(file)
            return jsonify({"result": result})

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)