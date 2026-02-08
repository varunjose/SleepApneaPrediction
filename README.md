# Deep Learning of Facial Depth Maps for Obstructive Sleep Apnea Prediction

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview

This project implements a **deep learning model for predicting Obstructive Sleep Apnea (OSA)** using facial depth maps. By analyzing facial features and their 3D structure, the model leverages advanced neural networks to identify subtle indicators of sleep apnea, providing an innovative, non-invasive method for early detection and diagnosis of the condition.

### 🎯 Key Innovation

Traditional OSA diagnosis requires expensive polysomnography tests in sleep labs. This project explores using **facial morphology as a biomarker** for OSA, potentially enabling:
- Early screening and detection
- Cost-effective preliminary assessment
- Non-invasive diagnostic support
- Accessible healthcare solutions

## ✨ Features

- 🧠 **VGG-19 Transfer Learning**: Utilizes pre-trained VGG-19 architecture for robust feature extraction
- 🎨 **Custom CNN Layers**: Additional convolutional layers fine-tuned for facial depth map analysis
- 🖥️ **Interactive GUI**: User-friendly Tkinter-based interface for seamless operation
- 🔮 **Real-time Prediction**: Instant OSA prediction from uploaded facial images
- 📊 **Visual Analytics**: Comprehensive training metrics visualization
- 💾 **Model Persistence**: Automatic saving and loading of trained models
- 🔄 **Data Preprocessing**: Automated image normalization and augmentation

## 🏗️ Model Architecture

The model combines **transfer learning** with custom convolutional layers:

```
Input Image (64×64×3)
    ↓
VGG-19 (Pre-trained, Frozen)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (256 units, ReLU)
    ↓
Dropout
    ↓
Dense (2 units, Softmax)
    ↓
Output: [OSA Detected / No OSA Detected]
```

## 📁 Project Structure

```
SleepApneaPrediction/
│
├── SleepApneaPrediction.py    # Main application file with GUI
├── README.md                   # Project documentation
├── SCREENS.docx               # Screenshots and documentation
├── run.bat                    # Windows batch file to run the application
│
├── Dataset/                   # Training dataset directory
│   ├── OSA Detected/         # Facial images of OSA patients
│   └── No OSA Detected/      # Facial images of healthy individuals
│
├── model/                     # Saved models and preprocessed data
│   ├── model.json            # Model architecture (JSON)
│   ├── model_weights.h5      # Trained model weights
│   ├── history.pckl          # Training history (accuracy, loss)
│   ├── X.npy                 # Preprocessed image data
│   └── Y.npy                 # Preprocessed labels
│
└── testimages/                # Test images for prediction
    └── [sample test images]
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/varunjose/SleepApneaPrediction.git
   cd SleepApneaPrediction
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

### Quick Start

**Windows Users:**
```bash
run.bat
```

**Mac/Linux Users:**
```bash
python SleepApneaPrediction.py
```

## 📖 Usage Guide

### 1️⃣ Upload Dataset

- Click **"Upload OSH Faces Dataset"**
- Select the `Dataset` folder containing subdirectories:
  - `OSA Detected/` - Images of patients diagnosed with OSA
  - `No OSA Detected/` - Images of healthy individuals
- Dataset structure is automatically validated

### 2️⃣ Preprocess Data

- Click **"Preprocess Dataset"**
- The system will:
  - Load all images from the dataset
  - Resize images to 64×64 pixels
  - Normalize pixel values to [0, 1]
  - Shuffle data for better training
  - Save preprocessed data to `model/X.npy` and `model/Y.npy`
- A sample processed image will be displayed

### 3️⃣ Train the Model

- Click **"Build VGG-19 Model"**
- If no pre-trained model exists:
  - VGG-19 base model is loaded with ImageNet weights
  - Custom layers are added and compiled
  - Training begins (10 epochs, batch size 16)
  - Model and weights are automatically saved
- If model exists, it loads from saved files
- Training accuracy is displayed upon completion

### 4️⃣ Make Predictions

- Click **"Upload Test Data & Predict OSH"**
- Select a facial depth map image from `testimages/`
- The model will:
  - Preprocess the image
  - Generate prediction with confidence
  - Display the result overlaid on the image
- Results show: **"OSA Detected"** or **"No OSA Detected"**

### 5️⃣ View Performance Metrics

- Click **"Accuracy Comparison Graph"**
- Visualize:
  - Training accuracy over epochs (green line)
  - Training loss over epochs (blue line)
  - Model convergence patterns

## 🔬 Technical Specifications

### Image Processing Pipeline
- **Input Format**: RGB images (any size)
- **Preprocessing**: Resize to 64×64, normalize to [0, 1]
- **Color Space**: RGB (3 channels)
- **Data Type**: float32

### Training Configuration
| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Categorical Cross-entropy |
| **Batch Size** | 16 |
| **Epochs** | 10 |
| **Learning Rate** | Default (0.001) |
| **Validation Split** | None (can be added) |

### Model Parameters
- **Total Parameters**: ~15M+ (including VGG-19)
- **Trainable Parameters**: ~1M (custom layers only)
- **Non-trainable Parameters**: ~14M (frozen VGG-19)

## 📊 Performance Metrics

The model's performance is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Loss**: Categorical cross-entropy loss
- **Visual Inspection**: Per-image prediction confidence

*Note: Actual performance depends on dataset quality and size*

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Processor**: Dual-core CPU

### Recommended Requirements
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 5GB+ free space
- **Processor**: Quad-core CPU

## 🛠️ Dependencies

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
opencv-python>=4.5.0
tensorflow>=2.4.0
Pillow>=8.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Use Cases

1. **Medical Screening**: Preliminary OSA screening in clinical settings
2. **Research**: Academic research on facial biomarkers for sleep disorders
3. **Health Apps**: Integration into telemedicine platforms
4. **Education**: Teaching deep learning and medical AI applications

## ⚠️ Important Disclaimers

> **Medical Disclaimer**: This software is designed for **research and educational purposes only**. It is NOT intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for OSA diagnosis and treatment.

- Results should be validated by medical professionals
- Not a replacement for polysomnography or clinical evaluation
- Model accuracy depends on training data quality
- Intended as a screening tool, not diagnostic device

## 🔮 Future Enhancements

- [ ] **Data Augmentation**: Implement rotation, flipping, and brightness adjustments
- [ ] **Model Improvements**: Test ResNet, EfficientNet, and custom architectures
- [ ] **Cross-Validation**: Implement K-fold validation for robust evaluation
- [ ] **Severity Classification**: Multi-class prediction (mild, moderate, severe OSA)
- [ ] **Web Application**: Deploy as a web service using Flask/FastAPI
- [ ] **Mobile App**: Develop iOS/Android applications
- [ ] **Explainability**: Add Grad-CAM visualization for model interpretability
- [ ] **Multi-Modal Input**: Integrate clinical metadata (age, BMI, neck circumference)
- [ ] **Real-time Detection**: Video-based real-time OSA risk assessment
- [ ] **Dataset Expansion**: Collect and train on larger, more diverse datasets

## 🤝 Contributing

Contributions are highly welcome! Here's how you can contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Ideas
- Improve model architecture
- Add more evaluation metrics
- Enhance GUI design
- Write unit tests
- Improve documentation
- Add new features

## 🐛 Known Issues

- Model training may be slow on CPU-only systems
- Large datasets require significant memory
- GUI may not scale properly on high-DPI displays

Report issues at: [GitHub Issues](https://github.com/varunjose/SleepApneaPrediction/issues)

## 📚 References & Research

This project is inspired by research on:
- Facial morphology as biomarkers for OSA
- Deep learning applications in medical imaging
- Transfer learning in healthcare AI

### Relevant Publications
- OSA detection using craniofacial features
- VGG-19 architecture and transfer learning
- Non-invasive sleep disorder screening methods


## 👨‍💻 Author

**Varun Jose**
- GitHub: [@varunjose](https://github.com/varunjose)

## 🙏 Acknowledgments

- **VGG Team** (Oxford) for the VGG-19 architecture
- **TensorFlow/Keras** teams for the deep learning framework
- **OpenCV** community for computer vision tools
- All contributors and researchers in medical AI

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/varunjose/SleepApneaPrediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/varunjose/SleepApneaPrediction/discussions)
- **Email**: Create an issue for correspondence

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ for advancing medical AI research

</div>
