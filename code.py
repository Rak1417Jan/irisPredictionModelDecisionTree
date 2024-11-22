import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
data

data.fillna(data.mean(), inplace=True)
data
# Count the occurrences of each class (Setosa, Versicolor, Virginica)
flower_counts = data['target'].value_counts()
flower_labels = ['Setosa', 'Versicolor', 'Virginica']

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    flower_counts,
    labels=flower_labels,
    autopct='%1.1f%%',  # Display percentage with 1 decimal place
    startangle=140,     # Start angle for better visualization
    colors=['#FF9999', '#66B2FF', '#99FF99'],  # Optional custom colors
    explode=(0.1, 0, 0) # Slightly separate the Setosa slice
)

# Add a title
plt.title('Percentage Distribution of Iris Flower Classes')
plt.show()


X = data.iloc[:, :-1]  # Features
y = data['target']     # Labels

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
scaler
X_normalized


X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

new_data = [[5.1, 3.5, 1.4, 0.2]]  # Modify as needed
new_data_normalized = scaler.transform(new_data)

prediction = model.predict(new_data_normalized)
predicted_class = iris.target_names[prediction[0]]
print(f"Predicted Class: {predicted_class}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example new data
new_data = [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.4, 1.4],
    [7.2, 3.6, 6.1, 2.5],
    [5.9, 3.0, 5.1, 1.8],
    [4.9, 3.0, 1.4, 0.2],
]


new_data_normalized = scaler.transform(new_data)


new_predictions = model.predict(new_data_normalized)


prediction_counts = pd.Series(new_predictions).value_counts(sort=False)
prediction_labels = ['Setosa', 'Versicolor', 'Virginica']


counts = [prediction_counts.get(i, 0) for i in range(3)]  


# Plot a horizontal bar graph for the predicted flower distribution
plt.figure(figsize=(8, 5))

# Create the bar graph
plt.barh(prediction_labels, counts, color=['#FF9999', '#66B2FF', '#99FF99'])

# Add labels and title
plt.xlabel('Count')
plt.ylabel('Flower Class')
plt.title('Distribution of Predicted Flower Classes')
plt.grid(axis='x', alpha=0.3)  # Add gridlines for the x-axis

# Show the plot
plt.show()
