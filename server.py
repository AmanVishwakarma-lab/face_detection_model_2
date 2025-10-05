from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import pickle
from io import BytesIO
from PIL import Image

import os

app = Flask(__name__)
CORS(app)

# Load embeddings
with open("embeddings/arcface_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)  # { "aman": [embedding1, ...], ... }

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    try:
        # Load image in memory
        img = Image.open(BytesIO(file.read()))
        img_path = BytesIO()
        img.save(img_path, format='JPEG')
        img_path.seek(0)

        # Detect all faces
        reps = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )

        recognized_names = []

        # Compare embeddings
        for face in reps:
            embedding = np.array(face["embedding"]).flatten()
            best_match_name = None
            best_score = -1
            for name, embeddings in known_faces.items():
                for known_embedding in embeddings:
                    sim = cosine_similarity(embedding, known_embedding)
                    if sim > best_score:
                        best_score = sim
                        best_match_name = name
            if best_score > 0.6:
                recognized_names.append(best_match_name)

        return jsonify({"recognized_faces": list(set(recognized_names))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ping")
def ping():
    return "API is live!"
@app.route("/")
def home():
    return "Welcome to the Face Detection API!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="https://facedetectionmodel2-production.up.railway.app/recognize", port=port, debug=False)




# it works well but it needed uploads folder

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from deepface import DeepFace
# import numpy as np
# import pickle
# import os

# app = Flask(__name__)
# CORS(app)

# # Load embeddings
# with open("embeddings/arcface_encodings.pkl", "rb") as f:
#     known_faces = pickle.load(f)  # { "aman": [embedding1, embedding2, ...], ... }

# def cosine_similarity(a, b):
#     a = np.array(a).flatten()
#     b = np.array(b).flatten()
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# @app.route("/recognize", methods=["POST"])
# def recognize():
#     if "image" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["image"]
#     filepath = os.path.join("uploads", file.filename)
#     os.makedirs("uploads", exist_ok=True)
#     file.save(filepath)

#     try:
#         # Detect all faces
#         reps = DeepFace.represent(
#             img_path=filepath,
#             model_name="ArcFace",
#             detector_backend="retinaface",  # best for multi-face
#             enforce_detection=False
#         )

#         recognized_names = []

#         # Loop over each detected face
#         for face in reps:
#             embedding = np.array(face["embedding"]).flatten()
#             best_match_name = None
#             best_score = -1

#             # Compare with all embeddings for each person
#             for name, embeddings in known_faces.items():
#                 for known_embedding in embeddings:
#                     sim = cosine_similarity(embedding, known_embedding)
#                     if sim > best_score:
#                         best_score = sim
#                         best_match_name = name

#             # Accept only if similarity above threshold
#             if best_score > 0.6:
#                 recognized_names.append(best_match_name)

#         #to remove the image 
#         os.remove(filepath)

#         return jsonify({"recognized_faces": list(set(recognized_names))})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host="0.0.0.0", port=port, debug=True)





