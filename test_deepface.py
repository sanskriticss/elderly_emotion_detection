import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time

def detect_emotions_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get the original video's FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / original_fps)  # Delay in milliseconds between frames
    
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
                    
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    
                    face_roi = frame[y:y+h, x:x+w]
                    
                    try:
                        result = DeepFace.analyze(face_roi, 
                                                actions=['emotion'],
                                                enforce_detection=False)
                        
                        emotions = result[0]['emotion']
                        dominant_emotion = result[0]['dominant_emotion']
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        label = f"{dominant_emotion}: {emotions[dominant_emotion]:.1f}%"
                        label_y = max(y - 10, 20)
                        
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame,
                                    (x, label_y - text_height - 5),
                                    (x + text_width, label_y + 5),
                                    (0, 255, 0),
                                    -1)
                        
                        cv2.putText(frame, label,
                                   (x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (0, 0, 0), 2)
                        
                    except Exception as e:
                        print(f"Error analyzing emotion: {str(e)}")
        
        cv2.imshow('Emotion Detection', frame)
        
        # Wait based on original video's FPS
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_video("v2.mp4")  # or 0 for webcam