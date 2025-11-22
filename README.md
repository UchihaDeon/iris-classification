# 🌸 Iris Flower Classification Project

## 📖 Overview
This project implements a **supervised machine learning pipeline** to classify Iris flowers into three species:  
- *Setosa*  
- *Versicolor*  
- *Virginica*  

It demonstrates the full workflow of a beginner-friendly ML project: **data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment via a Streamlit GUI**.  
The goal is to provide a modular, transparent, and professional structure that can be reused for academic submissions or client-ready deliverables.

---

## 📂 Project Structure


---

## 🚀 Features
- **Data Preprocessing**: Modular functions to load and prepare features/labels  
- **Exploratory Data Analysis (EDA)**: Visual insights into feature distributions and class separability  
- **Model Training**: KNN and Decision Tree classifiers with reproducible train/test splits  
- **Evaluation**: Accuracy, confusion matrix, precision, recall, and F1-score  
- **Interactive GUI**: Streamlit app for real-time classification and performance visualization  
- **Dataset Preview**: Explore the Iris dataset directly inside the app  

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/iris-classification.git
   cd iris-classification
   

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
**Usage**

**Run End-to-End Pipeline**
   ```bash
   python main.py
```

**Launch Streamlit GUI**
   ```bash
   python -m streamlit run app.py
```

## 📊 OutputsModel

**Model Performance**
- Accuracy: ~95% (depending on train/test split)
- Classification Report: Precision, recall, F1-score per class
- Confusion Matrix Heatmap: Visual evaluation of predictions

**Streamlit Dashboard**

- Input sepal/petal measurements → get predicted species
- View accuracy, classification report, and confusion matrix interactively
- Preview dataset samples inside the app

## 🛠️ Tech Stack

- Language: Python 3.13
- Libraries: Pandas, Scikit-learn, Seaborn, Matplotlib
- Deployment: Streamlit

## 🌟 Future Enhancements

- Add more classifiers (SVM, Random Forest)
- Deploy Streamlit app online (Streamlit Cloud / Heroku)
- Enable CSV upload for batch predictions
- Implement tabbed layout for cleaner GUI navigation
## 👨‍💻 Author

Developed by Deon — BCA undergraduate, full-stack developer, and data science intern.
Focused on building modular, client-ready machine learning projects with professional documentation and reproducible workflows.
