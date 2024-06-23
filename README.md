# Emotion Detection using OpenCV and DeepFace

This project demonstrates real-time emotion detection from a webcam feed using OpenCV and DeepFace. It detects faces in the video stream, predicts the dominant emotion for each detected face, and visualizes the emotions on a 2D graph.

## Features

- Detects faces using Haar Cascade Classifier.
- Analyzes emotions (angry, disgust, fear, happy, sad, surprise, neutral) using DeepFace.
- Visualizes emotions on a 2D graph based on valence and intensity.
- Provides real-time feedback on detected emotions overlaid on the video feed.

## Dependencies

- Python 3.x
- OpenCV
- Matplotlib
- NumPy
- DeepFace

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   cd emotion-detection
   ```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download Haar Cascade file:**
Download haarcascade_frontalface_default.xml from [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the project directory.

4. **Run the application**:
```bash
python emotion_detector.py
```

# Usage
- Upon running the application, a window will open showing the webcam feed with detected faces and their dominant emotions labeled.
- The 2D graph on the right side of the window will visualize the recent emotions detected over time.
- Close the application by pressing q in the video feed window.

# License
This project is licensed under the MIT License - see the LICENSE file for details.