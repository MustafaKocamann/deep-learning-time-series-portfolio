# Deep Learning Time Series Portfolio

## 📊 Executive Summary

This repository represents a comprehensive exploration of time series analysis methodologies, demonstrating end-to-end competency from fundamental preprocessing techniques to state-of-the-art deep learning architectures. The collection spans the entire machine learning lifecycle—data preprocessing, exploratory analysis, feature engineering, model development, and production-ready implementations.

The portfolio showcases practical applications across diverse domains: air quality forecasting, anomaly detection in industrial systems, fraud detection with transformers, earthquake early warning systems, human activity recognition, sensor error detection, traffic volume prediction, and electricity consumption estimation. Each project emphasizes not only achieving competitive performance metrics but also delivering interpretable, production-grade solutions that balance theoretical rigor with industrial pragmatism.

**Core Value Proposition:** A structured collection of real-world time series projects demonstrating mastery of classical statistical models through modern transformer-based architectures, with hands-on implementations addressing challenges like concept drift, missing data handling, multi-horizon forecasting, and computational efficiency.

---

## 🗺️ Project Overview & Categorization

### **Foundational Tier** 
*Classical forecasting and sequential modeling with LSTM/GRU*

| Project | Domain | Key Techniques | Tech Stack |
|---------|--------|----------------|------------|
| [Air Quality Estimation](air%20quality%20estimation/) | Environmental Monitoring | LSTM, Time Series Regression | PyTorch, NumPy |
| [Traffic Volume Estimation](traffic%20volume%20estimation/) | Transportation | GRU, Feature Engineering, Deployment (Flask/Streamlit) | PyTorch, Pandas, Scikit-learn |
| [Sensor Error Detection](sensor%20error%20detection/) | Industrial IoT | LSTM-FCN Hybrid, Binary Classification | PyTorch, Custom Architecture |

**Learning Outcomes:**
- Master LSTM/GRU architectures for sequential prediction
- Implement sliding window techniques for time series preprocessing
- Handle missing values and outliers in sensor data
- Deploy models with web interfaces

---

### **Advanced/Architectural Tier**
*Transformer architectures, autoencoders, and multi-modal systems*

| Project | Domain | Key Techniques | Tech Stack |
|---------|--------|----------------|------------|
| [Transformer Electricity Consumption](transformer%20electricity%20consumption%20estimation/) | Energy Management | Transformer Architecture, Multi-horizon Forecasting | PyTorch, Custom Transformers |
| [Detecting Fraud with Transformer](detecting%20fraud%20with%20transformer/) | Financial Security | Attention Mechanisms, Sequence Classification | PyTorch, Synthetic Data Generation |
| [Anomaly Detection](anomaly%20detection/) | System Monitoring | LSTM Autoencoder, Reconstruction Error | TensorFlow/Keras, Unsupervised Learning |
| [Earthquake Early Warning System](earthquake%20early%20warning%20system/) | Seismology | CNN for Time Series, Signal Processing | TensorFlow/Keras, Real-time Prediction |
| [Human Activity Detection](human%20activity%20detection/) | Wearable Computing | InceptionTime, Multi-sensor Fusion | TensorFlow/Keras, UCI HAR Dataset |

**Learning Outcomes:**
- Implement self-attention mechanisms for temporal modeling
- Design autoencoder architectures for anomaly detection
- Apply CNNs to time series classification tasks
- Work with multi-variate sensor data from accelerometers/gyroscopes
- Generate synthetic time series data for training

---

## 🛠️ Technical Ecosystem

### Core Technologies

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Libraries & Frameworks

| Category | Tools | Purpose |
|----------|-------|---------|
| **Deep Learning** | `PyTorch`, `TensorFlow/Keras`, `PyTorch Lightning` | Model development, training |
| **Data Processing** | `Pandas`, `NumPy`, `Scikit-learn` | Preprocessing, feature engineering |
| **Visualization** | `Matplotlib`, `Seaborn` | Time series plots, EDA |
| **Deployment** | `Flask`, `Streamlit`, `Joblib` | Model serving, web apps |
| **Architectures** | Custom Transformers, LSTM-FCN, InceptionTime, Autoencoders | Specialized time series models |

---

## 📁 Repository Structure

```
deep-learning-time-series-portfolio/
│
├── 📂 air quality estimation/
│   ├── AirQualityUCI.csv          # Dataset
│   ├── preprocessing.py            # Data cleaning & feature engineering
│   ├── eda.py                      # Exploratory data analysis
│   ├── train.py                    # LSTM training pipeline
│   ├── test.py                     # Model evaluation
│   └── lstm_model.pt               # Trained model weights
│
├── 📂 traffic volume estimation/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── model.py                    # GRU architecture
│   ├── preprocessing.py            # Time-based features
│   ├── train.py                    # Training loop
│   ├── test.py                     # Inference
│   ├── app.py                      # 🚀 Streamlit/Flask deployment
│   └── best_gru_model.pth          # Saved model
│
├── 📂 transformer electricity consumption estimation/
│   ├── household_power_consumption.txt
│   ├── transformer.py              # Custom Transformer implementation
│   ├── data.py                     # Data loader utilities
│   ├── preprocessing.py            # Sequence generation
│   ├── train.py                    # Multi-horizon training
│   └── transformer_energy_model.pth
│
├── 📂 detecting fraud with transformer/
│   ├── generate_data.py            # Synthetic transaction data
│   ├── transformer_model.py        # Attention-based classifier
│   ├── preprocessing.py            # Feature scaling
│   ├── train.py                    # Binary classification training
│   └── transformer_fraud_model.pth
│
├── 📂 anomaly detection/
│   ├── sentetik_dizi.csv           # Synthetic time series
│   ├── preprocessing.py            # Normalization
│   ├── train.py                    # LSTM Autoencoder training
│   ├── anomaly.py                  # Detection algorithm
│   └── lstm_autoencoder.h5         # Keras model
│
├── 📂 earthquake early warning system/
│   ├── data/                       # Seismic signal datasets
│   ├── preprocessing.py            # Signal preprocessing
│   ├── train.py                    # CNN training
│   ├── test.py                     # Real-time prediction simulation
│   ├── visualize.py                # Waveform visualization
│   └── cnn_model.h5
│
├── 📂 human activity detection/
│   ├── UCI HAR Dataset/            # Raw accelerometer/gyroscope data
│   ├── preprocessing.py            # Multi-sensor fusion
│   ├── segment.py                  # Time window segmentation
│   ├── inception.py                # InceptionTime architecture
│   ├── train.py                    # Multi-class classification
│   └── best_model.h5
│
├── 📂 sensor error detection/
│   ├── preprocessing.py            # SECOM dataset handling
│   ├── model.py                    # LSTM-FCN hybrid
│   ├── train.py                    # Binary classification
│   └── lstmfcn_secom.pth
│
└── 📄 README.md                    # This file
```

---

## 🎯 Key Highlights

### 1. **Diverse Architecture Portfolio**
- **Recurrent Networks**: LSTM, GRU, Bidirectional architectures
- **Convolutional Approaches**: 1D CNN for signal processing, InceptionTime
- **Attention Mechanisms**: Custom Transformer implementations
- **Hybrid Models**: LSTM-FCN combining recurrent and convolutional features
- **Autoencoders**: Unsupervised anomaly detection

### 2. **Production-Ready Code**
Each project includes:
- ✅ Modular preprocessing pipelines
- ✅ Comprehensive training scripts with checkpointing
- ✅ Evaluation metrics (MAE, RMSE, F1-Score, AUC)
- ✅ Saved model artifacts (.pt, .h5, scalers)
- ✅ Deployment examples (see `traffic volume estimation/app.py`)

### 3. **Real-World Applications**
- **Environmental**: Air quality forecasting for public health
- **Industrial**: Sensor error detection in manufacturing
- **Financial**: Fraud detection in transaction sequences
- **Public Safety**: Earthquake early warning systems
- **Smart Cities**: Traffic volume prediction for infrastructure planning
- **Energy**: Household power consumption optimization

### 4. **Dataset Diversity**
- Public benchmarks (UCI HAR, Air Quality UCI)
- Real-world data (Metro Traffic Volume, Household Power Consumption)
- Synthetic generation for specialized tasks (fraud detection, anomalies)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python >= 3.8
pip or conda
```

### Installation
```bash
# Clone the repository
git clone https://github.com/MustafaKocamann/deep-learning-time-series-portfolio.git
cd deep-learning-time-series-portfolio

# Install dependencies for a specific project
cd "air quality estimation"
pip install -r requirements.txt  # If requirements.txt exists

# Or install common dependencies
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn
pip install tensorflow  # For Keras-based projects
```

### Running a Project
```bash
# Example: Traffic Volume Estimation
cd "traffic volume estimation"
python preprocessing.py      # Prepare data
python train.py              # Train GRU model
python test.py               # Evaluate
python app.py                # Launch web app
```

---

## 📊 Performance Benchmarks

| Project | Metric | Score | Notes |
|---------|--------|-------|-------|
| Air Quality Estimation | MAE | *See test.py* | Multi-pollutant forecasting |
| Traffic Volume | RMSE | *See test.py* | Peak hour prediction |
| Fraud Detection | F1-Score | *See test.py* | Imbalanced dataset handling |
| Human Activity | Accuracy | *See test.py* | 6-class classification (UCI HAR) |
| Anomaly Detection | Precision@K | *See anomaly.py* | Reconstruction-based detection |

---

## 📚 Learning Resources

### Papers Implemented
- **Transformers**: "Attention Is All You Need" (Vaswani et al., 2017)
- **InceptionTime**: "InceptionTime: Finding AlexNet for Time Series Classification" (Fawaz et al., 2020)
- **LSTM-FCN**: "LSTM Fully Convolutional Networks for Time Series Classification" (Karim et al., 2019)

### Datasets Used
- [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- [UCI Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
- Metro Interstate Traffic Volume (Kaggle)

---

## 🤝 Contributing & Usage

This repository serves as:
- **Portfolio Showcase**: Demonstrating time series ML expertise
- **Learning Resource**: Reference implementations for common architectures
- **Baseline Code**: Starting points for similar projects

Feel free to:
- Fork and experiment with different architectures
- Report issues or suggest improvements
- Use code snippets with proper attribution

---

## 📄 License

This project is open source and available for educational purposes.

---

## 📧 Contact

**GitHub**: [@MustafaKocamann](https://github.com/MustafaKocamann)

**Maintainer**: Machine Learning Engineer specializing in time series analysis and deep learning  
**Last Updated**: January 2026  
**Status**: ✅ Active

---

## 🌟 Acknowledgments

- UCI Machine Learning Repository for public datasets
- PyTorch and TensorFlow communities for excellent documentation
- Research papers that inspired these implementations

