SOLUTION = """
# REAL-TIME FACE DETECTION WITH YUNET AND SFACE IDENTITY VERIFICATION

import cv2
import numpy as np
import os
import urllib.request
import matplotlib.pyplot as plt

# =============================================================================
# 1. Download Pre-trained YuNet Model (ONNX format)
# =============================================================================
model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
model_path = "face_detection_yunet.onnx"

if not os.path.exists(model_path):
    print("Downloading YuNet ONNX model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

yunet = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# =============================================================================
# 2. Static Image Demo (load from disk or use a generated test image)
# =============================================================================
# To test with a real face image: replace this path with your own image file.
# e.g. img = cv2.imread("my_photo.jpg")
# For demonstration we use a synthetic placeholder image.

# Attempt to load a test image; fall back to a blank placeholder if not found
test_image_path = "test_face.jpg"
if os.path.exists(test_image_path):
    img = cv2.imread(test_image_path)
    print(f"Loaded image: {test_image_path}")
else:
    print("No test image found. Using a 480x640 blank placeholder (no faces will be detected).")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

h, w, _ = img.shape
yunet.setInputSize((w, h))

# Detect faces
_, faces = yunet.detect(img)

# Draw bounding boxes and landmarks on a copy
annotated = img.copy()
if faces is not None:
    print(f"Detected {len(faces)} face(s).")
    for face in faces:
        box = list(map(int, face[:4]))
        cv2.rectangle(annotated, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        landmarks = list(map(int, face[4:14]))
        for i in range(5):
            cv2.circle(annotated, (landmarks[2*i], landmarks[2*i+1]), 3, (0, 0, 255), -1)
else:
    print("No faces detected in this image.")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("YuNet Face Detection")
plt.show()

# =============================================================================
# FACE ALIGNMENT AND EXTRACTION (AFFINE TRANSFORMATION)
# =============================================================================

standard_landmarks = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(image, face_data):
    detected_landmarks = np.array(face_data[4:14]).reshape(5, 2).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(detected_landmarks, standard_landmarks)
    aligned_face = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
    return aligned_face

if faces is not None and len(faces) > 0:
    target_face = faces[0]
    cropped_aligned_face = align_face(img, target_face)
    display_img = cv2.cvtColor(cropped_aligned_face, cv2.COLOR_BGR2RGB)

    print("Face aligned and extracted successfully. Shape:", cropped_aligned_face.shape)

    plt.figure(figsize=(3, 3))
    plt.imshow(display_img)
    plt.axis('off')
    plt.title("Aligned Face (112x112)")
    plt.show()
else:
    print("No faces to align. Skipping alignment step.")
    cropped_aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)

# =============================================================================
# FACE EMBEDDING EXTRACTION & IDENTITY VERIFICATION
# =============================================================================

recognizer_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
recognizer_path = "face_recognition_sface.onnx"

if not os.path.exists(recognizer_path):
    print("Downloading SFace ONNX Recognition model...")
    urllib.request.urlretrieve(recognizer_url, recognizer_path)
    print("Download complete.\\n")

face_recognizer = cv2.FaceRecognizerSF.create(
    model=recognizer_path,
    config="",
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

print("Extracting facial features...")
user_embedding = face_recognizer.feature(cropped_aligned_face)

print(f"Embedding generated! Shape: {user_embedding.shape}")
print(f"First 5 values: {user_embedding[0][:5]}\\n")

def calculate_cosine_similarity(feature1, feature2):
    score = cv2.FaceRecognizerSF.match(
        face_recognizer, feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE
    )
    return score

print("=" * 45)
print("IDENTITY VERIFICATION TESTS")
print("=" * 45)

score_self = calculate_cosine_similarity(user_embedding, user_embedding)
print(f"Test A (Self vs Self)        : {score_self:.4f} (Perfect Match)")

fake_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
fake_embedding = face_recognizer.feature(fake_face)
score_fake = calculate_cosine_similarity(user_embedding, fake_embedding)
print(f"Test B (Self vs Random Noise): {score_fake:.4f} (Different Identity)")
print("=" * 45)
print("Standard SFace Threshold: >= 0.363 indicates the same person.")
""".strip()

DESCRIPTION = "Detect faces with YuNet, align them via affine transformation, and verify identity using SFace 128-D embeddings and cosine similarity."
