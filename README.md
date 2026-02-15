# 🧠 Automated Brain Stroke Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered web application that analyzes CT scan images to detect the risk of brain stroke. This system utilizes an **ensemble of Deep Learning models (ResNet50 & EfficientNetB0)** to provide accurate predictions and includes **Grad-CAM explainability** to visualize the regions of interest in the brain.

## 🚀 Live Demo
**[Launch App](https://automated-brain-stroke-detection-system.streamlit.app/)**

## ✨ Features

-   **Deep Learning Ensemble**: Combines the power of **ResNet50** and **EfficientNetB0** for robust classification.
-   **Instant Analysis**: Upload a CT scan (JPG/PNG) and get results in seconds.
-   **Explainable AI (XAI)**: Visualizes **Grad-CAM heatmaps** to show exactly which parts of the image influenced the model's decision.
-   **Interactive Controls**:
    -   Adjust **Decision Thresholds** dynamically.
    -   Tune **Ensemble Weights** between models.
    -   Toggle heatmap smoothing and model selection.
-   **User-Friendly Interface**: Clean, responsive UI built with Streamlit.

## 🛠️ Tech Stack

-   **Frontend & UI**: [Streamlit](https://streamlit.io/)
-   **Deep Learning Framework**: [TensorFlow / Keras](https://www.tensorflow.org/)
-   **Image Processing**: OpenCV, Pillow
-   **Data Visualization**: Matplotlib

## ⚙️ Installation & Local Run

To run this project locally, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Nuthan-27/automated-brain-stroke-detection-system.git
    cd automated-brain-stroke-detection-system
    ```

2.  **Install Git LFS (Important)**
    Since the model files are large, you need Git Large File Storage.
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install Dependencies**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**
    ```bash
    streamlit run app/streamlit_app.py
    ```

## 📂 Project Structure

```
├── app/
│   ├── streamlit_app.py   # Main application logic
│   ├── utils.py           # Helper functions
│   └── requirements.txt   # App-specific dependencies
├── models/
│   ├── resnet50_best.h5       # Trained ResNet50 model
│   └── efficientnetb0_best.h5 # Trained EfficientNetB0 model
├── requirements.txt       # Project dependencies
├── packages.txt           # System-level dependencies
└── README.md              # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License.
