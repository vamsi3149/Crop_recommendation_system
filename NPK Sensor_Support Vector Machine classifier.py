import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import the Support Vector Machine classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the data into features (temperature, humidity, N, P, K) and target (recommended crop)
X = data[['temperature', 'humidity', 'N', 'P', 'K']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Support Vector Machine (SVM) classifier
clf = SVC(random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)


# Calculate the accuracy of the predictions
accuracy = np.mean(y_pred == y_test)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('F1 Score:', f1)

# Now, you can use the trained model to recommend the best crop based on input data
# Replace 'new_data' with the actual data (Temperature, Humidity, N, P, K) you obtain
new_data = pd.DataFrame({'temperature': [28], 'humidity': [70], 'N': [70], 'P': [50], 'K': [40]})

recommended_crop = clf.predict(new_data)

print(f"Recommended Crop: {recommended_crop[0]}")
