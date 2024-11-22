import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Fill any missing values
data.fillna(data.mean(), inplace=True)

# Prepare features and target
X = data.iloc[:, :-1]  # Features
y = data['target']     # Labels

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Iris Flower Classification")
st.markdown("""
This app classifies Iris flowers based on the sepal and petal dimensions you provide.
It predicts the flower class and displays a bar chart showing the count of each predicted flower type.
""")

# Input Section
st.header("Input Flower Data")
st.markdown("Enter sepal and petal dimensions for one or more flowers.")
user_input = st.text_area("Input array (e.g., [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4]])", 
                          value="[[5.1, 3.5, 1.4, 0.2]]")

try:
    # Convert input to array
    new_data = np.array(eval(user_input))

    # Normalize the input data
    new_data_normalized = scaler.transform(new_data)

    # Predict flower classes
    predictions = model.predict(new_data_normalized)
    predicted_classes = [iris.target_names[p] for p in predictions]

    # Display predictions
    st.subheader("Predicted Flower Classes")
    st.write(predicted_classes)

    # Calculate and plot distribution
    prediction_counts = pd.Series(predictions).value_counts(sort=False)
    counts = [prediction_counts.get(i, 0) for i in range(3)]
    prediction_labels = ['Setosa', 'Versicolor', 'Virginica']

    # Bar Graph
    st.subheader("Prediction Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(prediction_labels, counts, color=['#FF9999', '#66B2FF', '#99FF99'])
    ax.set_xlabel("Count")
    ax.set_ylabel("Flower Class")
    ax.set_title("Distribution of Predicted Flower Classes")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error in input processing: {e}")

# Display model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.header("Model Information")
st.sidebar.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
