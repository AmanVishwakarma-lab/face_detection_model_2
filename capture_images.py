import cv2
import os
# Number of images per person
NUM_IMAGES = 50
SAVE_PATH = "known_faces"

# Ask for person's name
person_name = input("Enter the person's name: ").strip()

# Create folder if not exists
person_folder = os.path.join(SAVE_PATH, person_name)
os.makedirs(person_folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)                                    # enable it for laptop camera
# cap = cv2.VideoCapture("http://192.0.0.4:8080/video")      #enable it for ip webcam

print(f"📸 Capturing {NUM_IMAGES} images for {person_name}...")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to capture image")
        break

    # Show video feed
    cv2.imshow("Capture - Press SPACE to save, ESC to exit", frame)

    key = cv2.waitKey(1)

    # Press SPACE to capture an image
    if key == 32:  # SPACE key
        count += 1
        img_path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved {img_path}")

        if count >= NUM_IMAGES:
            print(" Done capturing!")
            break

    # Press ESC to quit early
    elif key == 27:  # ESC key
        print("Capture aborted by user")
        break

cap.release()
cv2.destroyAllWindows()
