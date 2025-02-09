# Face Emotion Recognition

## Overview
This project is a deep learning-based **Face Emotion Recognition** system that classifies facial expressions into seven categories: **angry, disgust, fear, happy, neutral, sad, and surprise**. The model is trained using **CNN and Transfer Learning (MobileNetV2)** to compare their performance. The best-performing model is then used in a **Streamlit web application** for real-time predictions.

## Web Interface 
![Screenshot 2025-02-09 142143](https://github.com/user-attachments/assets/b1284725-82d8-484b-8f65-028b7ef7b321)


## Features
- **Deep Learning Model**: Built using **CNN and MobileNetV2** for emotion recognition.
- **Data Preprocessing**: Resized images, handled class imbalance, and performed data augmentation.
- **Model Training & Evaluation**: Compared multiple models and tuned hyperparameters for best accuracy.
- **Streamlit Web App**: Allows users to upload images and get real-time emotion predictions.
- **Deployment Ready**: Model saved in `.keras` format for easy loading and use in applications.

## Dataset
The dataset consists of facial images categorized into seven emotion labels:
- **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, **surprise**

The dataset is divided into:
- **Training Set**
- **Validation Set**

Data Source:
- Collected from publicly available facial expression databases and crowd-sourced contributions.

Purpose:
- To train and evaluate the DeepFER model for accurate and real-time facial emotion recognition across diverse scenarios.

## Model Architectures
### 1️⃣ **CNN Model**
A custom **Convolutional Neural Network (CNN)** with:
- Multiple **Conv2D layers**
- **Batch Normalization**, **MaxPooling** and **Dropout** for regularization
- **ReLU activations** and **Softmax** output for classification

### 2️⃣ **Transfer Learning with MobileNetV2**
- Pretrained **MobileNetV2** model (trained on ImageNet)
- Fine-tuned last 20 layers
- Added **fully connected layers**, **batch normalization**, and **dropout** for improved performance
- Loss calculated by **sparse_categorical_crossentropy** and **Accuracy** metrics used for evaluation 

## Model Performance
- **CNN Model Accuracy**: ~28%
- **MobileNetV2 Accuracy**: ~63% (Best Model)

| Metric      | CNN Model | MobileNetV2 |
|------------|-----------|--------------|
| Accuracy   | 28%       | 63%          |
| Loss       | 1.74      | 1.04         |

## Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/ish-war/Face-Emotion-Recognition
cd Face-Emotion-Recognition
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Optional)
```sh
python experiments.py
```

### 4️⃣ Run the Streamlit App
```sh
streamlit run app.py
```

## Usage
1. **Upload a facial image** through the Streamlit interface.
2. The model predicts and displays the corresponding emotion.
3. Results are shown with **real-time image processing**.

## File Structure
```
📂 face-emotion-recognition
│── 📂 data/                   # Dataset folder
│── 📂 models/                 # Saved models (.keras)
│── 📂 notebooks/              # Jupyter notebooks for training & EDA
│── app.py                     # Streamlit app for emotion detection
│── experiments.py              # Model training script
│── requirements.txt            # Python dependencies
│── README.md                   # Project Documentation
```

## Future Improvements
- ✅ Implement **real-time emotion detection** via webcam.
- ✅ Improve accuracy with **larger datasets** and more augmentation.
  
## Contributors
- **Ishwar Ambad** - https://github.com/ish-war/Face-Emotion-Recognition

## License
This project is licensed under the **MIT License**.

