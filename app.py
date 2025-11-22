import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessing import load_data, prepare_features_labels
from src.model import train_knn, train_decision_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title
st.title("🌸 Iris Flower Classification Dashboard")

# Sidebar for model choice
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose a model:", ["KNN", "Decision Tree"])

# Input fields for prediction
st.subheader("🔮 Classify a Flower")
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Load and prepare data
df = load_data("data/iris.csv")
X, y = prepare_features_labels(df)

# Train chosen model
if model_choice == "KNN":
    model, X_test, y_test = train_knn(X, y, k=5)
else:
    model, X_test, y_test = train_decision_tree(X, y)

# Prediction section
if st.button("Classify"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    st.success(f"🌸 Predicted Species: **{prediction}**")

# Evaluation section
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.2f}")

# Classification report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix heatmap
st.subheader("🔎 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Dataset preview
st.subheader("📂 Dataset Preview")
if st.checkbox("Show Iris Dataset"):
    st.dataframe(df.head(10))