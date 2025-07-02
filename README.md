# Facial-Emotion-Detection

A deep learning project that detects human emotions from facial expressions using a Convolutional Neural Network (CNN) with both downsampling and upsampling blocks. It classifies emotions from static images and live webcam feed using the FER2013 dataset.

---

## Emotion Classes

The model recognizes the following **7 facial emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## Repository Overview

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `codefile.ipynb`              | Jupyter notebook for training the CNN model     |
| `realtimedetection.py`        | Real-time facial emotion detection using webcam |
| `facial-emotion-detector.json`| Saved model architecture                        |
| `facial-emotion-detector.h5`  | Trained model weights                           |
| `real-time-demo.pdf`          | Screenshots and results from real-time inference|
| `README.md`                   | Project documentation                           |

---

## Model Architecture

This CNN architecture includes **downsampling via Conv2D + MaxPooling**, followed by **upsampling using Conv2DTranspose**, and finally a **classification head**.
Input: (48, 48, 1)

Downsampling
Conv2D(64) → MaxPool → Dropout
Conv2D(128) → MaxPool → Dropout
Conv2D(256) → MaxPool → Dropout

Upsampling
Conv2DTranspose(128) → 6×6 → 12×12
Conv2DTranspose(64) → 12×12 → 24×24
Conv2DTranspose(32) → 24×24 → 48×48

Output
GlobalAveragePooling2D → Dense(128) → Dropout → Dense(7, softmax)

- **Loss Function:** `categorical_crossentropy`  
- **Optimizer:** `Adam`  
- **Activations:** `ReLU` in hidden layers, `Softmax` in output

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HemakshiKumar/Facial-Emotion-Detection.git
cd Facial-Emotion-Detection
```
### 2. Install Requirements
```bash
pip install numpy pandas matplotlib tensorflow keras opencv-python
```

## Real-Time Emotion Detection

Run the real-time detection script:

```bash
python realtimedetection.py
```

It will:

- Access your webcam
- Detect faces using Haar cascades
- Classify facial emotions live
- Overlay the predicted emotion on the video frame

---

## Possible Improvements
- Add data augmentation to improve generalization
- Use more advanced face detectors (e.g., MTCNN or Dlib)
- Try transfer learning using pretrained models like ResNet or MobileNet

### Deploy the model via Flask/Streamlit as a web app

## License
This project is for academic and educational purposes.

### Author
Hemakshi Kumar
Feel free to contribute, open issues, or star ⭐ the repo!



