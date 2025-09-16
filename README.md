# 🤖 Machine Learning Projects Collection

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-blue)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green)](https://flask.palletsprojects.com/)

Welcome to my comprehensive collection of machine learning projects! This repository showcases diverse applications of machine learning across different domains including healthcare, finance, computer vision, and natural language processing.

## 📚 Table of Contents

1. [Project Overview](#-project-overview)
2. [Technologies Used](#-technologies-used)
3. [Projects](#-projects)
4. [Installation & Setup](#-installation--setup)
5. [Usage](#-usage)
6. [Project Structure](#-project-structure)
7. [Contributing](#-contributing)
8. [Contact](#-contact)

## 🔍 Project Overview

This repository contains 10+ machine learning projects covering:
- **Healthcare Predictions**: Diabetes and Heart Disease Prediction
- **Financial Modeling**: Loan Status and Gold Price Prediction
- **Human Resources**: Employee Attrition Analysis
- **Computer Vision**: Image Classification with CNNs
- **Natural Language Processing**: Sentiment Analysis and POS Tagging
- **Model Optimization**: Hyperparameter Tuning Techniques
- **Web Deployment**: Flask-based ML Applications

## 🛠 Technologies Used

### Programming Languages & Frameworks
- **Python 3.7+** - Primary programming language
- **Jupyter Notebooks** - Interactive development environment
- **Flask** - Web application framework

### Machine Learning Libraries
- **Scikit-Learn** - Traditional ML algorithms
- **TensorFlow/Keras** - Deep learning models
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Seaborn/Matplotlib** - Data visualization

### Additional Tools
- **MongoDB** - Database for web applications
- **Keras Tuner** - Hyperparameter optimization
- **NLTK** - Natural language processing
- **OpenCV** - Computer vision tasks

## 🚀 Projects

### 1. 🩺 Diabetes Prediction
**Directory**: `Project 1: Diabetes_prediction/`

**Description**: Developed a machine learning model to predict diabetes risk using patient data. The project implements various preprocessing techniques including quantile transformation for handling skewed data distributions.

**Key Features**:
- Data preprocessing with QuantileTransformer
- Logistic Regression implementation
- Model persistence with joblib
- Comprehensive data analysis and visualization

**Technologies**: Python, Scikit-Learn, Pandas, Seaborn, Logistic Regression

**Files**:
- `Diabetes_prediction.ipynb` - Main analysis notebook
- `diabetes.csv` - Dataset

---

### 2. 💰 Loan Status Prediction
**Directory**: `Project 2: Loan_Status_Prediction/`

**Description**: Built a predictive model to determine loan approval status based on applicant information. The project utilizes Support Vector Machine (SVM) classifier with comprehensive data preprocessing.

**Key Features**:
- Data cleaning and feature engineering
- Label encoding for categorical variables
- SVM with linear kernel implementation
- Cross-validation and model evaluation

**Technologies**: Python, Scikit-Learn, SVM, Pandas, NumPy

**Files**:
- `Loan_Status_Prediction_Sikandar.ipynb` - Complete analysis
- `train_u6lujuX_CVtuZ9i (1).csv` - Training dataset

---

### 3. 👥 Employee Attrition Prediction
**Directory**: `Project 3: Employee_Attrition_Prediction/`

**Description**: Designed a machine learning model to forecast employee turnover using logistic regression. The model analyzes various employee factors to predict potential attrition.

**Key Features**:
- HR analytics and data exploration
- Logistic regression modeling
- Feature importance analysis
- Retention strategy insights

**Technologies**: Python, Scikit-Learn, Logistic Regression, Pandas

**Files**:
- `LogisticRegression(Employee_Attrition_Prediction).ipynb` - Analysis notebook
- `HR_comma_sep.csv` - Employee dataset

---

### 4. 🏆 Gold Price Prediction
**Directory**: `Project 4: Gold Price Prediction/`

**Description**: Constructed a time series forecasting model to predict gold prices using historical data. The project employs various statistical and machine learning methods for accurate price prediction.

**Key Features**:
- Time series analysis
- Feature engineering with financial indicators
- Multiple regression techniques
- Investment insights generation

**Technologies**: Python, Scikit-Learn, Time Series Analysis, Pandas

**Files**:
- `Gold_Price_Prediction.ipynb` - Prediction model
- `gld_price_data.csv` - Historical gold price data

---

### 5. ⚙️ Hyperparameter Tuning & Deployment
**Directory**: `Project 5: Hyperparameter Tunning/`

**Description**: Comprehensive project showcasing hyperparameter optimization techniques using Keras Tuner for heart disease prediction, complete with Flask web application deployment.

**Key Features**:
- Keras Tuner implementation for ANN optimization
- Flask web application with MongoDB integration
- User registration and prediction system
- Model deployment with real-time predictions
- HTML templates for web interface

**Technologies**: Python, TensorFlow/Keras, Keras Tuner, Flask, MongoDB, HTML/CSS

**Files**:
- `Keras Tunner ANN(heart_disease_prediction).ipynb` - Hyperparameter tuning
- `app.py` - Flask web application
- `heart_disease_pred_model.h5` - Trained model
- `scaler.pkl` - Data preprocessing scaler
- HTML templates (`index.html`, `register.html`, `result.html`)
- `Heart Disease Prediction Website.mp4` - Demo video

---

### 6. 🐱🐶 Cat-Dog Classifier CNN
**Directory**: `Project 6: Cat Dog Classifier CNN/`

**Description**: Implemented a Convolutional Neural Network (CNN) to classify images of cats and dogs. The project demonstrates advanced computer vision techniques with deep learning.

**Key Features**:
- CNN architecture design
- Image preprocessing and augmentation
- Transfer learning techniques
- Real-time image classification

**Technologies**: Python, TensorFlow/Keras, CNN, Computer Vision

**Files**:
- `cat-dog-classifier-cnn.ipynb` - CNN implementation

---

### 7. 🖼️ CIFAR-10 CNN Model
**Directory**: `Project 7: Cifar_10_CNN_Model/`

**Description**: Developed a CNN model to classify images from the CIFAR-10 dataset, which includes ten different object classes. Applied advanced regularization techniques for improved generalization.

**Key Features**:
- Multi-class image classification
- Data augmentation strategies
- Regularization techniques (Dropout, L2)
- Model evaluation and visualization

**Technologies**: Python, TensorFlow/Keras, CNN, CIFAR-10 Dataset

**Files**:
- `Cifar_10_CNN.ipynb` - CIFAR-10 classification model

---

### 8. 💭 Sentiment Analysis
**Directory**: `Project 8: sentiment analysis/`

**Description**: Created a comprehensive NLP pipeline to perform sentiment analysis on text data. The project implements various text preprocessing techniques and classification algorithms.

**Key Features**:
- Text preprocessing (tokenization, stemming, lemmatization)
- Feature engineering with NLP techniques
- Sentiment classification (positive/negative/neutral)
- One-hot encoding for text data

**Technologies**: Python, NLTK, Scikit-Learn, NLP, Text Classification

**Files**:
- `sentiment_analysis.ipynb` - Complete NLP pipeline

---

### 9. ❤️ ANN Heart Disease Prediction
**File**: `ANN(heart_disease_prediction).ipynb`

**Description**: Developed an Artificial Neural Network (ANN) to predict heart disease risk based on patient health metrics. This project focuses on deep learning approaches for medical diagnosis.

**Key Features**:
- Neural network architecture design
- Medical data preprocessing
- Feature selection and scaling
- Early detection capabilities

**Technologies**: Python, TensorFlow/Keras, ANN, Medical Data Analysis

---

### 10. 📝 Natural Language Processing Collection
**Directory**: `NLP/`

**Description**: A comprehensive collection of NLP projects covering various text processing and analysis tasks.

**Projects Included**:

#### 10.1 Parts of Speech Tagging
- **File**: `parts_of_speech_tagging.ipynb`
- **Description**: NLP model for accurate POS tagging in sentences
- **Applications**: Linguistic analysis, language processing

#### 10.2 Text Classification
- **File**: `text-classification.ipynb`
- **Description**: Multi-class text classification system
- **Applications**: Document categorization, content filtering

#### 10.3 Next Word Predictor RNN
- **File**: `next-word-predictor-rnn.ipynb`
- **Description**: RNN-based model for predicting next words in sequences
- **Applications**: Auto-completion, text generation

#### 10.4 CIFAR-10 CNN (NLP Version)
- **File**: `cifar-10-cnn.ipynb`
- **Description**: Alternative implementation of CIFAR-10 classification
- **Applications**: Image recognition, computer vision

**Technologies**: Python, NLTK, TensorFlow/Keras, RNN, NLP

## 📦 Installation & Setup

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
pip package manager
```

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/Sikandarh11/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

2. **Create virtual environment** (recommended):
```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

3. **Install required packages**:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
pip install jupyter notebook flask pymongo nltk opencv-python
pip install keras-tuner joblib bcrypt
```

4. **Additional setup for NLP projects**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🎯 Usage

### Running Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to the desired project directory and open the `.ipynb` files.

### Running Flask Web Application
```bash
cd "Project 5: Hyperparameter Tunning"
python app.py
```
Access the application at `http://localhost:5000`

### Working with Individual Projects
Each project directory contains:
- Jupyter notebook with complete analysis
- Dataset files
- Additional resources (models, scalers, etc.)

## 📁 Project Structure

```
Machine-Learning-Projects/
├── README.md
├── ANN(heart_disease_prediction).ipynb
├── Project 1: Diabetes_prediction/
│   ├── Diabetes_prediction.ipynb
│   └── diabetes.csv
├── Project 2: Loan_Status_Prediction/
│   ├── Loan_Status_Prediction_Sikandar.ipynb
│   └── train_u6lujuX_CVtuZ9i (1).csv
├── Project 3: Employee_Attrition_Prediction/
│   ├── LogisticRegression(Employee_Attrition_Prediction).ipynb
│   └── HR_comma_sep.csv
├── Project 4: Gold Price Prediction/
│   ├── Gold_Price_Prediction.ipynb
│   └── gld_price_data.csv
├── Project 5: Hyperparameter Tunning/
│   ├── Keras Tunner ANN(heart_disease_prediction).ipynb
│   ├── app.py
│   ├── heart_disease_pred_model.h5
│   ├── scaler.pkl
│   ├── index.html
│   ├── register.html
│   ├── result.html
│   └── Heart Disease Prediction Website.mp4
├── Project 6: Cat Dog Classifier CNN/
│   └── cat-dog-classifier-cnn.ipynb
├── Project 7: Cifar_10_CNN_Model/
│   └── Cifar_10_CNN.ipynb
├── Project 8: sentiment analysis/
│   └── sentiment_analysis.ipynb
└── NLP/
    ├── parts_of_speech_tagging.ipynb
    ├── text-classification.ipynb
    ├── next-word-predictor-rnn.ipynb
    └── cifar-10-cnn.ipynb
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Guidelines:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Sikandar Hassan**
- GitHub: [@Sikandarh11](https://github.com/Sikandarh11)
- Project Link: [https://github.com/Sikandarh11/Machine-Learning-Projects](https://github.com/Sikandarh11/Machine-Learning-Projects)

---

⭐ **If you found this repository helpful, please give it a star!** ⭐

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*This repository represents a comprehensive journey through various machine learning domains, showcasing practical implementations and real-world applications. Each project is designed to demonstrate different aspects of machine learning, from basic algorithms to advanced deep learning techniques and web deployment.*
