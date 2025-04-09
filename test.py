import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

def preprocess_face(face_img, target_size=(48, 48)):
    """
    Preprocess the face image according to model requirements
    Modify this function based on how your model was trained
    """
    # Convert to grayscale if your model expects grayscale input
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to match your model's input size
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values if your model expects normalized input
    normalized = resized / 255.0
    
    # Reshape for model input (adjust dimensions based on your model)
    preprocessed = np.expand_dims(np.expand_dims(normalized, -1), 0)
    return preprocessed

def detect_emotions_video(video_source=0, model_path='path_to_your_model.h5'):
    # Load your pre-trained model
    model = load_model(model_path)
    
    # Define emotion labels (adjust these to match your model's classes)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get original FPS for proper video speed
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / original_fps)
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)
                    
                    # Ensure coordinates are within frame
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    try:
                        # Preprocess face
                        preprocessed_face = preprocess_face(face_roi)
                        
                        # Get prediction from your model
                        predictions = model.predict(preprocessed_face, verbose=0)
                        
                        # Get dominant emotion
                        emotion_idx = np.argmax(predictions[0])
                        emotion_label = emotions[emotion_idx]
                        confidence = predictions[0][emotion_idx] * 100
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Prepare label
                        label = f"{emotion_label}: {confidence:.1f}%"
                        label_y = max(y - 10, 20)
                        
                        # Add background rectangle for text
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame,
                                    (x, label_y - text_height - 5),
                                    (x + text_width, label_y + 5),
                                    (0, 255, 0),
                                    -1)
                        
                        # Add text
                        cv2.putText(frame, label,
                                   (x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (0, 0, 0), 2)
                        
                    except Exception as e:
                        print(f"Error in emotion detection: {str(e)}")
        
        cv2.imshow('Emotion Detection', frame)
        
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_video(
        "v1.mp4",  # or 0 for webcam
        "emotion_model.h5"  # replace with your model path
    )