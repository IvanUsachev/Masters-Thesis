import os
import numpy as np
import cv2 as cv
from insightface.app import FaceAnalysis

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.embeddings = []
        self.app = FaceAnalysis(name='buffalo_l', ctx_id=0)  
        self.app.prepare(ctx_id=0)  

    def extract_face_and_embedding(self, img):
        faces = self.app.get(img)
        if faces:
            face = faces[0]  
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            face_img = img[y1:y2, x1:x2]
            face_resized = cv.resize(face_img, self.target_size)
            return face_resized, face.embedding
        else:
            return None, None

    def load_faces_and_embeddings(self, video_path):
        cap = cv.VideoCapture(video_path)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, num=30, dtype=np.int32)

        extracted_faces = []
        embeddings = []
        for idx in frame_indices:
            cap.set(cv.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                face, embedding = self.extract_face_and_embedding(frame)
                if face is not None:
                    extracted_faces.append(face)
                    embeddings.append(embedding)
        
        cap.release()
        return extracted_faces, embeddings

    def load_classes(self):
        for video_name in os.listdir(self.directory):
            base_name = os.path.splitext(video_name)[0]
            video_path = os.path.join(self.directory, video_name)
            faces, embeddings = self.load_faces_and_embeddings(video_path)
            labels = [base_name for _ in range(len(faces))]
            print(f"Loaded successfully: {len(labels)} frames from {video_name}")
            self.X.extend(faces)
            self.Y.extend(labels)
            self.embeddings.extend(embeddings)
        
        return np.asarray(self.X), np.asarray(self.Y), np.asarray(self.embeddings)