import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from datetime import datetime, timedelta
from deepface import DeepFace

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class EmotionDetector:
    def __init__(self):
        # Initialize the face cascade classifier and emotion mapping
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_map = {
            'angry': (-0.5, 0.8, 'red'),
            'disgust': (-0.6, 0.6, 'purple'),
            'fear': (-0.7, 0.7, 'gray'),
            'happy': (0.7, 0.8, 'yellow'),
            'sad': (-0.8, 0.5, 'blue'),
            'surprise': (0.5, 0.7, 'orange'),
            'neutral': (0.0, 0.2, 'green')
        }
        self.path = deque(maxlen=100)  # To store the path of the ball

    def detect_faces_emotions(self, frame):
        """
        Detect faces and their dominant emotions in the given frame.

        Parameters:
        frame (numpy.ndarray): Frame from the webcam feed.

        Returns:
        numpy.ndarray: Frame with rectangles around detected faces and emotion labels.
        list: List of tuples containing detected emotion and face coordinates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detections = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(results, list) and results:
                emotion = results[0]['dominant_emotion']
                detections.append((emotion, (x, y, w, h)))
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        return frame, detections

    def update_graph(self, frame, detections):
        """
        Update the 2D graph with recent emotion data.

        Parameters:
        frame (numpy.ndarray): Frame from the webcam feed.
        detections (list): List of tuples containing detected emotion and face coordinates.
        """
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Valence')
        ax.set_ylabel('Intensity')

        if not detections:
            return
        
        legend_handles = []
        for emotion, (x, y, w, h) in detections:
            if emotion in self.emotion_map:
                valence, intensity, color = self.emotion_map[emotion]
                self.path.append((datetime.now(), valence, intensity, color))
                legend_handles.append(ax.scatter([], [], c=color, s=100, label=emotion))
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        current_time = datetime.now()
        recent_path = [(v, i, c) for t, v, i, c in self.path if current_time - t <= timedelta(seconds=10)]

        if recent_path:
            for i in range(1, len(recent_path)):
                valence1, intensity1, color1 = recent_path[i-1]
                valence2, intensity2, color2 = recent_path[i]
                ax.plot([valence1, valence2], [intensity1, intensity2], color=color1, alpha=0.6)

            valence, intensity, color = recent_path[-1]
            ax.scatter(valence, intensity, c=color, s=100)
        
        ax.legend(handles=legend_handles, loc='upper right', fontsize='small')
        fig.canvas.draw()
        fig.canvas.flush_events()

def main():
    # Specify the index of your external webcam
    webcam_index = 1

    cap = cv2.VideoCapture(webcam_index)
    detector = EmotionDetector()

    if not cap.isOpened():
        print(f"Error: Could not open video stream from webcam {webcam_index}.")
        return

    plt.ion()
    global fig, ax
    fig, ax = plt.subplots()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, detections = detector.detect_faces_emotions(frame)
            detector.update_graph(frame, detections)

            cv2.imshow('Face and Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
