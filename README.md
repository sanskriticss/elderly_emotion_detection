Elderly Emotion Detection
=========================

This repository provides a framework for detecting emotions in elderly individuals using facial images and video data. The project leverages deep learning techniques to analyze facial expressions and classify them into corresponding emotional states.

Features
--------

*   **Facial Emotion Recognition**: Utilizes deep learning models to identify and classify emotions from facial images.
    
*   **Video Analysis**: Processes video data to detect and track emotional expressions over time.
    
*   **Pre-trained Models**: Incorporates pre-trained models for efficient and accurate emotion detection.
    

Installation
------------

To set up the environment for this project, follow these steps:

1. git clone https://github.com/sanskriticss/elderly\_emotion\_detection.git
    
2. cd elderly\_emotion\_detection
    
3.  pip install tensorflow keras opencv-python deepface
    

Usage
-----

The project includes several scripts for training models and testing emotion detection:

*   **train.py**: Script for training the emotion detection model.
    
*   **test.py**: Script for testing the model on sample data.
    
*   **test_deepface.py**: Demonstrates the use of the DeepFace library for emotion recognition.
    
*   **test_img.py**: Processes and analyzes individual images for emotion detection.
    

### Training the Model

To train the model, run:

` python train.py   `

Ensure that your training data is organized and accessible as required by the script.

### Testing the Model



### Using DeepFace for Emotion Detection

The test_deepface.py script demonstrates how to use the [DeepFace](https://github.com/serengil/deepface) library for emotion recognition:

`   python test_deepface.py   `

DeepFace is a Python library for deep learning-based face recognition and facial attribute analysis. It can detect emotions such as happy, sad, angry, and surprised from facial images.


Data
----

The repository includes sample video files (v1.mp4, v2.mp4) that can be used for testing the emotion detection capabilities. Ensure you have the appropriate permissions to use these files. And training is done on the following dataset:
https://drive.google.com/drive/folders/1Bzm6ebN_2a-GQovUtAuwOYu2cnKwyIJs?usp=sharing
