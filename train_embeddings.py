from deepface import DeepFace
import os
import pickle
import numpy as np

DATASET_PATH = "known_faces"
SAVE_PATH = "embeddings/arcface_encodings.pkl"

known_faces = {}

print("🔄 Training embeddings with DeepFace (ArcFace backend)...")

# Loop over each person
for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        try:
            # Generate embedding for each image
            rep = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            embeddings.append(rep[0]["embedding"])
        except Exception as e:
            print(f"⚠️ Skipping {img_name}: {e}")

    if len(embeddings) > 0:
        known_faces[person] = embeddings  # store list of embeddings
        print(f"✅ Processed {person}: {len(embeddings)} images")

# Save embeddings
os.makedirs("embeddings", exist_ok=True)
with open(SAVE_PATH, "wb") as f:
    pickle.dump(known_faces, f)

print(f"🎉 Training complete! Saved to {SAVE_PATH}")







