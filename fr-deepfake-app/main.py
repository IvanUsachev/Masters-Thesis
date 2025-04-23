import os
from flask import Flask, request, render_template
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import threading
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import timm
import csv
from datetime import datetime
from faceloading import FACELOADING
from flask import send_file
import tempfile
import shutil


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----- Load Models -----
app = Flask(__name__)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('best_vit_model.pth', map_location=device))
model.to(device)
model.eval()

# Load existing embeddings if available
known_embeddings = []
known_labels = []

if os.path.exists('faces_embeddings_done.npz'):
    with np.load('faces_embeddings_done.npz') as data:
        known_embeddings = data['EMBEDDED_X']
        known_labels = data['Y']

def preprocess_face_image(face_img):
    pil_image = Image.fromarray(face_img)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    return input_tensor

def recognize_face(embedding, known_embeddings, known_labels, face_threshold=0.4):
    distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]
    min_distance = min(distances)
    if min_distance < face_threshold:
        index = distances.index(min_distance)
        return known_labels[index]
    return 'Unknown'

def predict_face(input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        fake_prob = probabilities[0][0].item()
    return fake_prob

def process_video(video_path, dfd_threshold=0.2, face_threshold=0.4, fake_threshold=0.25, detect_deepfake=True, display=True):
    cap = cv.VideoCapture(video_path)
    video_filename = "Webcam Stream" if isinstance(video_path, int) else os.path.basename(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_interval = int(fps / 1)  # process one frame per second
    total_frames = 0
    fake_frames = 0
    real_frames = 0
    known_frames = 0
    unknown_frames = 0
    deepfake_results = []

    def deepfake_worker(frame, frame_number):
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)  
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_img = img_rgb[y1:y2, x1:x2]
            preprocessed_face = preprocess_face_image(face_img)
            prediction = predict_face(preprocessed_face)
            is_fake = prediction > dfd_threshold

            if is_fake:
                deepfake_results.append((frame_number, "Fake"))
            else:
                deepfake_results.append((frame_number, "Real"))

            #print(f"Frame {frame_number}: Deepfake detection: {'Fake' if is_fake else 'Real'} (Prediction: {prediction:.2f})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)  
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_img = img_rgb[y1:y2, x1:x2]
            embedding = face.embedding  
            label = recognize_face(embedding, known_embeddings, known_labels, face_threshold=face_threshold)
            
            if label == 'Unknown':
                unknown_frames += 1
            else:
                known_frames += 1
            
            # Perform deepfake detection on one frame per second
            if detect_deepfake and total_frames % frame_interval == 0 and label != 'Unknown':
                #and label != 'Unknown'
                deepfake_thread = threading.Thread(target=deepfake_worker, args=(frame, total_frames))
                deepfake_thread.start()
                deepfake_thread.join()

            # Determine Fake or Real based on collected results
            num_fake = sum(1 for _, status in deepfake_results if status == "Fake")
            num_real = sum(1 for _, status in deepfake_results if status == "Real")
            fake_percentage = num_fake / (num_fake + num_real) if (num_fake + num_real) > 0 else 0
            
            if fake_percentage > fake_threshold:
                label = "Fake"
                color = (255, 0, 0)  # Blue for Fake
            else:
                color = (0, 0, 255) if label == 'Unknown' else (0, 255, 0)  # Red for unknown, Green for known
            
            if display:
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if display:
            cv.putText(frame, f'Playing: {video_filename}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.imshow('Face Recognition', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv.destroyAllWindows()
    
    face_prediction = "Known" if known_frames >= unknown_frames else "Unknown"
    
    # Determine deepfake prediction if detection is enabled
    video_prediction = face_prediction  
    if detect_deepfake and deepfake_results:
        num_fake = sum(1 for _, status in deepfake_results if status == "Fake")
        num_real = sum(1 for _, status in deepfake_results if status == "Real")
        fake_percentage = num_fake / (num_fake + num_real) if (num_fake + num_real) > 0 else 0

        if fake_percentage > fake_threshold:
            video_prediction = "Fake"
        else:
            video_prediction = "Real"
    
    #final_prediction = face_prediction
    #final_prediction = video_prediction

    # Prioritize face recognition if no known face is found
    final_prediction = face_prediction if face_prediction == "Unknown" else video_prediction
    print(f"The video '{video_filename}' is classified as '{final_prediction}'")
    
    return final_prediction

# ----- Flask Routes -----
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    result_cam = None

    if request.method == "POST":
        if "video" in request.files:
            file = request.files["video"]
            if file and file.filename:
                filepath = os.path.join("uploads", file.filename)
                os.makedirs("uploads", exist_ok=True)
                file.save(filepath)
                result = process_video(filepath)
                os.remove(filepath)

    return render_template("index.html", result=result, result_cam=result_cam)

@app.route("/batch_test", methods=["GET", "POST"])
def batch_test():
    if request.method == "POST":
        uploaded_files = request.files.getlist("videos")
        if not uploaded_files:
            return "❌ No video files selected."

        temp_dir = tempfile.mkdtemp()
        results = []

        for file in uploaded_files:
            if file.filename == '':
                continue
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            prediction = process_video(file_path, display=False)
            results.append((file.filename, prediction))

        # Save results to CSV
        csv_path = os.path.join(temp_dir, "batch_results.csv")
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Filename", "Prediction"])
            writer.writerows(results)

        # Send file back for download with message
        filename = "batch_results.csv"
        web_path = os.path.join("results", filename)
        final_path = os.path.join("results", filename)

        os.makedirs("results", exist_ok=True)
        os.replace(csv_path, final_path)  # Move CSV to persistent location

        return render_template("batch_result.html", download_link=web_path, file_path=os.path.abspath(final_path))

    return render_template("batch_test.html")

@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    global known_embeddings, known_labels
    message = None

    if request.method == "POST":
        # Handle deletion
        if request.form.get("delete") == "true":
            if os.path.exists('faces_embeddings_done.npz'):
                os.remove('faces_embeddings_done.npz')
                known_embeddings = []
                known_labels = []
                message = "🗑️ Saved embeddings have been deleted."
            else:
                message = "⚠️ No embeddings file found to delete."

        # Handle enrollment
        else:
            uploaded_files = request.files.getlist("files")
            if not uploaded_files:
                message = "❌ No files uploaded."
            else:
                temp_dir = "temp_enroll_uploads"
                os.makedirs(temp_dir, exist_ok=True)

                for file in uploaded_files:
                    if file.filename:
                        file.save(os.path.join(temp_dir, file.filename))

                face_loader = FACELOADING(directory=temp_dir)
                X, Y, embeddings = face_loader.load_classes()

                np.savez_compressed('faces_embeddings_done.npz', EMBEDDED_X=embeddings, Y=Y)
                known_embeddings = embeddings
                known_labels = Y

                import shutil
                shutil.rmtree(temp_dir)

                count = len(uploaded_files)
                message = f"✅ Successfully enrolled from uploaded files — {count} file(s)"

    return render_template("enroll.html", message=message)

@app.route("/webcam", methods=["GET", "POST"])
def webcam():
    result = None
    result_cam = None

    if request.method == "POST":
        result_cam = process_video(0)

    return render_template("index.html", result=result, result_cam=result_cam)

if __name__ == "__main__":
    app.run(debug=False)