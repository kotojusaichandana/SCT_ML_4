HAND GESTURE RECOGNITION USING DEEP LEARNING
==========================================

PROJECT OVERVIEW
----------------
This project implements a Hand Gesture Recognition system using Deep Learning and Computer Vision. 
A Convolutional Neural Network (CNN) is trained on a hand gesture image dataset and the trained model 
is then used to recognize hand gestures in real time using a webcam.

The system captures live video input, preprocesses the hand region, and predicts the gesture using 
a trained TensorFlow/Keras model.


OBJECTIVES
----------
• To build a real-time hand gesture recognition system
• To train a CNN model for gesture classification
• To perform live gesture detection using a webcam
• To apply image preprocessing for better prediction accuracy


TECHNOLOGIES USED
-----------------
• Programming Language : Python
• Deep Learning        : TensorFlow, Keras
• Computer Vision      : OpenCV
• Data Handling        : NumPy
• Dataset              : Leap Motion Hand Gesture Dataset
• IDE                  : VS Code
• Platform             : Windows


DATASET DESCRIPTION
-------------------
The project uses the Leap Motion Hand Gesture Recognition Dataset.

Dataset Structure:
dataset/
└── leapGestRecog/
    ├── 00/
    │   ├── 01_palm/
    │   ├── 02_l/
    │   ├── 03_fist/
    │   ├── 04_fist_moved/
    │   ├── 05_thumb/
    │   ├── 06_index/
    │   ├── 07_ok/
    │   ├── 08_palm_moved/
    │   ├── 09_c/
    │   └── 10_down/

Each folder represents a unique hand gesture class.


PROJECT STRUCTURE
-----------------
Task04_Hand_Gesture_Recognition/
│
├── dataset/
│   └── leapGestRecog/
│
├── model/
│   └── gesture_model.h5
│
├── train.py
├── test_webcam.py
├── README.txt
└── requirements.txt


MODEL ARCHITECTURE
------------------
• Convolutional Neural Network (CNN)
• Conv2D + ReLU Activation
• MaxPooling Layers
• Flatten Layer
• Fully Connected Dense Layers
• Dropout for regularization
• Softmax output layer for classification


WORKING PRINCIPLE
-----------------
1. The CNN model is trained using labeled gesture images from the dataset.
2. Images are resized, normalized, and converted to grayscale.
3. After training, the model is saved as "gesture_model.h5".
4. During testing, webcam captures live video frames.
5. A Region of Interest (ROI) is extracted from each frame.
6. Image preprocessing is applied to match training data.
7. The trained model predicts the gesture in real time.
8. The predicted gesture label is displayed on the screen.


IMPORTANT NOTE ON DATASET AND WEBCAM
-----------------------------------
The training dataset consists of infrared images captured using Leap Motion sensors, 
whereas the testing phase uses visible-light webcam images.

Due to this difference, predictions may occasionally show reduced accuracy. 
To minimize this gap, image preprocessing techniques such as grayscale conversion, 
blurring, and thresholding have been applied during webcam testing.

This limitation is inherent to the dataset and does not affect the correctness 
or implementation quality of the project.


INSTALLATION AND EXECUTION
--------------------------
1. Install Python (version 3.9 recommended)
2. Install required libraries:
   pip install tensorflow opencv-python numpy scikit-learn
3. Train the model:
   python train.py
4. Run real-time gesture recognition:
   python test_webcam.py
5. Press 'Q' to exit the webcam window


RESULTS
-------
• Successfully trained a CNN model for hand gesture recognition
• Real-time webcam gesture detection implemented
• Model predictions displayed live on the screen
• Project demonstrates practical application of Deep Learning and Computer Vision


APPLICATIONS
------------
• Human-Computer Interaction
• Touchless interfaces
• Virtual Reality (VR) systems
• Robotics control
• Assistive technologies


CONCLUSION
----------
This project demonstrates an effective approach to hand gesture recognition using deep learning. 
Despite dataset modality differences, the system successfully performs real-time gesture prediction. 
The project highlights the importance of preprocessing and model training in computer vision tasks 
and serves as a strong foundation for advanced gesture-based interaction systems.
