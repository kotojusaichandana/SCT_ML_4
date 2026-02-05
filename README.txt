# ✋ Hand Gesture Recognition using Machine Learning

## 📌 Project Overview
This project focuses on building a Hand Gesture Recognition system using Machine Learning techniques.  
The goal is to recognize human hand gestures from images or webcam input and classify them into predefined gesture categories.

This project demonstrates how computer vision and ML can be combined to enable human–computer interaction, which is widely used in real-world applications such as touchless interfaces, assistive technologies, and smart systems.

This project was completed as **Task 4** under the SkillCraft Technology Internship Program.

## 🎯 Objectives
- Understand image-based data preprocessing
- Train a machine learning model for gesture classification
- Perform real-time hand gesture recognition using webcam input
- Build an end-to-end ML pipeline from data loading to prediction

## 🧠 Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn / TensorFlow (model-based)
- Streamlit
- VS Code

## 📂 Dataset Information
The dataset used for this project contains **40,000+ hand gesture images** across multiple gesture classes.

⚠️ Due to GitHub file size limitations, the dataset is **not included** in this repository.

### Dataset Source
- Provided by internship / external dataset (e.g., Kaggle or institutional source)

### Expected Folder Structure
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
Once the dataset is placed in this structure, the training and testing scripts will work without modification.

## ⚙️ Project Workflow
1. Dataset loading and preprocessing
2. Image resizing and normalization
3. Feature extraction / CNN-based learning
4. Model training and evaluation
5. Saving trained model
6. Real-time gesture prediction using webcam

## ▶️ How to Run the Project

### Step 1: Install dependencies
pip install numpy opencv-python scikit-learn streamlit

### Step 2: Train the model
python train.py

### Step 3: Test using webcam
python test_webcam.py

### Step 4 (Optional): Run UI
streamlit run app.py

## 📊 Results
- Successfully classified multiple hand gestures
- Achieved reliable real-time predictions
- Demonstrated practical ML application with live input

## 🚀 Key Learnings
- Working with large-scale image datasets
- Handling real-time computer vision pipelines
- Understanding challenges in gesture-based ML systems
- Improving model accuracy through experimentation

## 📌 Conclusion
This project strengthened my understanding of **Machine Learning + Computer Vision** and how theoretical concepts translate into real-world applications.
It reflects hands-on experience in building, debugging, and deploying an ML-based system.

## 👤 Author
**Sai Chandana Kotoju**  
SkillCraft Technology Internship – Task 4

## 📜 License
This project is for educational and internship evaluation purposes.



